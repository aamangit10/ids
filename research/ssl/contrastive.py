import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class InfoNCELoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) / InfoNCE Contrastive Loss.
    Encourages two augmented views of the same network flow to have high cosine similarity
    while pushing away different flows in the batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: L2 normalized projection tensor of shape (batch_size, proj_dim)
            z2: L2 normalized projection tensor of shape (batch_size, proj_dim)
        Returns:
            Scalar contrastive loss value.
        """
        batch_size = z1.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        # Concatenate projections: shape (2 * batch_size, proj_dim)
        z = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix: shape (2 * batch_size, 2 * batch_size)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create positive pair indices (i, i + batch_size) and (i + batch_size, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z1.device),
            torch.arange(0, batch_size, device=z1.device)
        ], dim=0)

        # Mask out self-similarity on diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


def apply_packet_sequence_augmentation(
    x: torch.Tensor,
    padding_mask: torch.Tensor,
    mask_prob: float = 0.15,
    jitter_std: float = 0.05,
    drop_pkt_prob: float = 0.10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies non-destructive packet-level contrastive augmentations:
    1. Feature Masking: randomly masks selected token features (replaces with 0).
    2. Feature Jittering: adds small Gaussian noise to continuous packet metrics (length, IAT).
    3. Token Dropping: randomly marks valid packet tokens as padded tokens.

    Args:
        x: Tensor of shape (batch_size, seq_len, feature_dim)
        padding_mask: BoolTensor of shape (batch_size, seq_len)
    """
    x_aug = x.clone()
    mask_aug = padding_mask.clone()

    batch_size, seq_len, feat_dim = x.shape

    # 1. Feature Jittering (add small noise to packet length and IAT)
    noise = torch.randn_like(x_aug) * jitter_std
    # Apply noise only to valid non-padded tokens
    valid_mask = (~mask_aug).unsqueeze(-1).float()
    x_aug = x_aug + (noise * valid_mask)

    # 2. Feature Masking (randomly zero out individual feature values)
    feat_mask = (torch.rand_like(x_aug) < mask_prob) & (valid_mask.bool())
    x_aug[feat_mask] = 0.0

    # 3. Token Dropping (randomly set packet token as padded)
    token_drop_mask = (torch.rand(batch_size, seq_len, device=x.device) < drop_pkt_prob) & (~padding_mask)
    mask_aug = mask_aug | token_drop_mask

    return x_aug, mask_aug
