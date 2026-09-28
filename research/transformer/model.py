import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for network packet sequences."""

    def __init__(self, embed_dim: int, max_seq_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with positional encodings added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class PacketTransformerEncoder(nn.Module):
    """
    Self-Supervised & Supervised PyTorch Transformer Encoder for Network Flow Intrusion Detection.
    Converts 5-feature packet sequence tokens -> dense embeddings -> Transformer -> classification head / projection head.
    """

    def __init__(
        self,
        features_dim: int = 5,
        embed_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        num_classes: int = 2,
        projection_dim: int = 128
    ):
        super().__init__()
        self.features_dim = features_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        # Packet feature projection embedding layer
        self.feature_projection = nn.Sequential(
            nn.Linear(features_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # SSL Projection Head (InfoNCE / NT-Xent contrastive loss)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )

        # Classification Head (Binary or Multiclass)
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def extract_flow_representation(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Passes packet features through embedding, positional encoding, and Transformer.
        Performs masked mean-pooling across the sequence dimension to obtain flow representation.

        Args:
            x: Packet features tensor of shape (batch_size, seq_len, features_dim)
            padding_mask: BoolTensor of shape (batch_size, seq_len), True for padded tokens.

        Returns:
            Flow embedding tensor of shape (batch_size, embed_dim)
        """
        if padding_mask is not None:
            if padding_mask.size(0) != x.size(0):
                padding_mask = padding_mask.expand(x.size(0), -1)

        # 1. Feature projection: (batch_size, seq_len, embed_dim)
        h = self.feature_projection(x)

        # 2. Add Positional Encoding
        h = self.pos_encoder(h)

        # 3. Transformer Encoder
        encoded = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        # 4. Masked Mean Pooling over valid packets
        if padding_mask is not None:
            # Mask shape: (batch_size, seq_len, 1)
            mask_inv = (~padding_mask).unsqueeze(-1).float()
            sum_encoded = torch.sum(encoded * mask_inv, dim=1)
            count = torch.clamp(mask_inv.sum(dim=1), min=1.0)
            flow_rep = sum_encoded / count
        else:
            flow_rep = torch.mean(encoded, dim=1)

        return flow_rep

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_projection: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        If return_projection is True, returns normalized L2 contrastive embeddings (for SSL pretraining).
        Otherwise returns classification logits (for fine-tuning / evaluation).
        """
        flow_rep = self.extract_flow_representation(x, padding_mask=padding_mask)

        if return_projection:
            proj = self.projection_head(flow_rep)
            return F.normalize(proj, dim=-1)
        else:
            logits = self.classification_head(flow_rep)
            return logits
