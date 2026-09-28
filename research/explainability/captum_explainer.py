import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


class TransformerCaptumExplainer:
    """
    Computes packet sequence feature attributions for PyTorch PacketTransformerEncoder using Captum Integrated Gradients.
    Provides packet-level timeline attributions and packet-feature attributions.
    """

    def __init__(self, model: torch.nn.Module, feature_names: Optional[List[str]] = None):
        self.model = model
        self.model.eval()
        self.feature_names = feature_names or ["Packet Length", "Protocol", "TCP Flags", "Inter-Arrival Time", "Direction"]

    def explain_sequence(
        self,
        sequence_tensor: torch.Tensor,
        padding_mask: torch.Tensor,
        target_class: int = 1,
        n_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Calculates Integrated Gradients attributions for a single packet flow sequence.

        Args:
            sequence_tensor: Tensor of shape (1, seq_len, feature_dim) or (seq_len, feature_dim)
            padding_mask: BoolTensor of shape (1, seq_len) or (seq_len,)
            target_class: 1 for ATTACK, 0 for BENIGN
            n_steps: Number of integration steps for Integrated Gradients

        Returns:
            Dictionary containing packet timeline attributions and feature attributions.
        """
        if sequence_tensor.ndim == 2:
            sequence_tensor = sequence_tensor.unsqueeze(0)
        if padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)

        device = next(self.model.parameters()).device
        x = sequence_tensor.to(device).requires_grad_(True)
        mask = padding_mask.to(device)

        seq_len = x.shape[1]
        feat_dim = x.shape[2]

        if CAPTUM_AVAILABLE:
            # Captum Integrated Gradients wrapper function
            def forward_func(inputs):
                return self.model(inputs, padding_mask=mask, return_projection=False)

            ig = IntegratedGradients(forward_func)
            baseline = torch.zeros_like(x)

            attributions, delta = ig.attribute(
                x,
                baselines=baseline,
                target=target_class,
                n_steps=n_steps,
                return_convergence_delta=True
            )
            attr_np = attributions.squeeze(0).detach().cpu().numpy()  # Shape: (seq_len, feature_dim)
        else:
            # Gradient x Input fallback when Captum is not installed
            logits = self.model(x, padding_mask=mask, return_projection=False)
            target_score = logits[0, target_class]
            target_score.backward()
            grad = x.grad.detach().cpu().numpy()[0]
            attr_np = grad * x.detach().cpu().numpy()[0]

        # Filter out padded tokens using mask
        mask_np = mask.squeeze(0).cpu().numpy()
        real_tokens_count = int((~mask_np).sum())

        packet_attributions = []
        for p_idx in range(real_tokens_count):
            feat_attrs = {}
            total_pkt_attr = 0.0

            for f_idx, feat_name in enumerate(self.feature_names):
                val = float(attr_np[p_idx, f_idx])
                feat_attrs[feat_name] = round(val, 4)
                total_pkt_attr += abs(val)

            packet_attributions.append({
                "packet_index": p_idx + 1,
                "total_importance": round(total_pkt_attr, 4),
                "feature_attributions": feat_attrs
            })

        # Calculate overall feature importance across the flow sequence
        overall_feature_importance = {}
        for f_idx, feat_name in enumerate(self.feature_names):
            mean_attr = float(np.mean(np.abs(attr_np[:real_tokens_count, f_idx]))) if real_tokens_count > 0 else 0.0
            overall_feature_importance[feat_name] = round(mean_attr, 4)

        return {
            "num_packets_analyzed": real_tokens_count,
            "target_class_explained": "MALICIOUS" if target_class == 1 else "BENIGN",
            "packet_timeline": packet_attributions,
            "overall_feature_importance": overall_feature_importance,
            "explanation_method": "Captum Integrated Gradients" if CAPTUM_AVAILABLE else "Gradient x Input Fallback",
            "disclaimer": "Integrated Gradients attributions reflect model internal feature weights and do not constitute causal proof of malicious intent."
        }
