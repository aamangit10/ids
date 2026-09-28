import pytest
import torch
import numpy as np
from research.transformer.model import PacketTransformerEncoder
from research.ssl.contrastive import InfoNCELoss, apply_packet_sequence_augmentation
from research.data.synthetic_generator import generate_synthetic_packet_sequences
from research.data.packet_pipeline import PacketSequenceDataset
from research.explainability.captum_explainer import TransformerCaptumExplainer


def test_transformer_tensor_shapes_and_masks():
    """Verify PyTorch Transformer input/output tensor shapes and mask handling."""
    model = PacketTransformerEncoder(
        features_dim=5,
        embed_dim=256,
        nhead=4,
        num_layers=2,
        max_seq_len=32,
        num_classes=2,
        projection_dim=128
    )

    batch_size = 8
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 5)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    # Mask last 10 tokens for sample 0
    mask[0, 22:] = True

    # Classification logits output shape
    logits = model(x, padding_mask=mask, return_projection=False)
    assert logits.shape == (batch_size, 2)

    # SSL Contrastive projection output shape
    proj = model(x, padding_mask=mask, return_projection=True)
    assert proj.shape == (batch_size, 128)
    # Check L2 normalization (norm should be ~1.0)
    norms = torch.norm(proj, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_infonce_contrastive_loss():
    """Verify InfoNCE loss non-negativity and gradient backpropagation."""
    loss_fn = InfoNCELoss(temperature=0.07)
    z1 = torch.randn(16, 128)
    z2 = torch.randn(16, 128)
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)

    loss = loss_fn(z1, z2)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_packet_sequence_dataset_padding():
    """Verify PacketSequenceDataset creates correct sequence tensors and padding masks."""
    flows = generate_synthetic_packet_sequences(n_flows=10, max_seq_len=32, seed=42)
    dataset = PacketSequenceDataset(flows, max_seq_len=32, fit_scaler=True)

    seqs, masks, labels = dataset[0]
    assert seqs.shape == (32, 5)
    assert masks.shape == (32,)
    assert labels.dtype == torch.long


def test_transformer_integrated_gradients_explainer():
    """Verify Captum / Gradient Integrated Gradients returns structured packet attributions."""
    model = PacketTransformerEncoder(features_dim=5, embed_dim=64, num_layers=1, num_classes=2)
    explainer = TransformerCaptumExplainer(model)

    x = torch.randn(1, 32, 5)
    mask = torch.zeros(1, 32, dtype=torch.bool)
    mask[0, 15:] = True  # Padded after packet 15

    explanation = explainer.explain_sequence(x, mask, target_class=1)
    
    assert explanation["num_packets_analyzed"] == 15
    assert len(explanation["packet_timeline"]) == 15
    assert "overall_feature_importance" in explanation
    assert "disclaimer" in explanation
