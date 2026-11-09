from __future__ import annotations

import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader
from research.transformer.model import PacketTransformerEncoder
from research.ssl.contrastive import InfoNCELoss, apply_packet_sequence_augmentation
from research.data.synthetic_generator import create_demo_data_if_missing, generate_synthetic_packet_sequences
from research.data.packet_pipeline import PacketSequenceDataset, get_packet_dataloaders


def pretrain_ssl_transformer(
    flows: List[Dict[str, Any]],
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    temperature: float = 0.07,
    save_path: str = "models/ssl_pretrained_transformer.pt",
    seed: int = 42
) -> Tuple[PacketTransformerEncoder, List[float]]:
    """
    Executes Self-Supervised Contrastive Pretraining on unlabeled flow sequences.
    1. Generates two augmented views per flow sequence (A1 and A2).
    2. Passes both through Transformer encoder + projection head.
    3. Computes InfoNCE contrastive loss and optimizes encoder representations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = PacketSequenceDataset(flows, max_seq_len=32, fit_scaler=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PacketTransformerEncoder(
        features_dim=5,
        embed_dim=256,
        nhead=4,
        num_layers=4,
        max_seq_len=32,
        num_classes=2,
        projection_dim=128
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = InfoNCELoss(temperature=temperature)

    model.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0

        for x, mask, _ in loader:
            # Generate two augmented views A1 and A2
            x1, mask1 = apply_packet_sequence_augmentation(x, mask, mask_prob=0.15, jitter_std=0.05)
            x2, mask2 = apply_packet_sequence_augmentation(x, mask, mask_prob=0.15, jitter_std=0.05)

            # Get L2-normalized projections
            z1 = model(x1, padding_mask=mask1, return_projection=True)
            z2 = model(x2, padding_mask=mask2, return_projection=True)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        loss_history.append(avg_loss)

    # Save pretrained weights
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_stats": dataset.scaler_stats,
        "epochs": epochs,
        "loss_history": loss_history
    }, save_path)

    return model, loss_history


if __name__ == "__main__":
    print("Executing Self-Supervised Contrastive Pretraining...")
    flows = generate_synthetic_packet_sequences(n_flows=600, seed=42)
    model, loss_hist = pretrain_ssl_transformer(flows, epochs=5, save_path="models/ssl_pretrained_transformer.pt")
    print(f"SSL Pretraining Complete. Final Loss: {loss_hist[-1]:.4f}")
    print("Saved model checkpoint to 'models/ssl_pretrained_transformer.pt'.")

