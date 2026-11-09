import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from research.data.synthetic_generator import generate_synthetic_packet_sequences


class PacketSequenceDataset(Dataset):
    """
    PyTorch Dataset for Packet Sequence Mode.
    Transforms raw 5-tuple flow packets into fixed-length tensor sequences with padding masks.
    """

    def __init__(
        self,
        flows: List[Dict[str, Any]],
        max_seq_len: int = 32,
        feature_dim: int = 5,
        fit_scaler: bool = True,
        scaler_stats: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.flows = flows
        
        # Feature columns per packet token: [length, protocol, flags, inter_arrival_time, direction]
        self.scaler_stats = scaler_stats or {}
        
        # Parse tensors
        self.sequences, self.padding_masks, self.labels = self._process_flows(fit_scaler)

    def _process_flows(self, fit_scaler: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_raw_tokens = []
        parsed_flows = []
        labels_list = []

        for flow in self.flows:
            pkts = flow.get("packets", [])
            label = flow.get("is_attack", 0)
            labels_list.append(label)

            token_matrix = []
            for p in pkts[:self.max_seq_len]:
                # Extract non-identifying packet features
                length = float(p.get("length", 0.0))
                protocol = float(p.get("protocol", 6))
                flags = float(p.get("flags", 0))
                iat = float(p.get("inter_arrival_time", 0.0))
                direction = float(p.get("direction", 0))

                tok = [length, protocol, flags, iat, direction]
                token_matrix.append(tok)
                all_raw_tokens.append(tok)

            parsed_flows.append(token_matrix)

        all_raw_arr = np.array(all_raw_tokens) if len(all_raw_tokens) > 0 else np.zeros((1, self.feature_dim))

        # Calculate mean & std for numerical normalization
        if fit_scaler:
            for feat_idx in range(self.feature_dim):
                mean_val = float(np.mean(all_raw_arr[:, feat_idx]))
                std_val = float(np.std(all_raw_arr[:, feat_idx])) + 1e-6
                self.scaler_stats[feat_idx] = (mean_val, std_val)

        # Normalize and pad sequences
        tensor_seqs = []
        tensor_masks = []

        for token_matrix in parsed_flows:
            seq_len = len(token_matrix)
            padded_seq = np.zeros((self.max_seq_len, self.feature_dim), dtype=np.float32)
            mask = np.ones(self.max_seq_len, dtype=bool)  # True indicates PADDING

            for t_idx, tok in enumerate(token_matrix):
                norm_tok = []
                for feat_idx, val in enumerate(tok):
                    m, s = self.scaler_stats.get(feat_idx, (0.0, 1.0))
                    norm_tok.append((val - m) / s)
                
                padded_seq[t_idx] = norm_tok
                mask[t_idx] = False  # False indicates REAL packet token

            tensor_seqs.append(padded_seq)
            tensor_masks.append(mask)

        return (
            torch.tensor(np.array(tensor_seqs), dtype=torch.float32),
            torch.tensor(np.array(tensor_masks), dtype=torch.bool),
            torch.tensor(labels_list, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sequence_tensor, padding_mask, label)."""
        return self.sequences[idx], self.padding_masks[idx], self.labels[idx]


def get_packet_dataloaders(
    flows: List[Dict[str, Any]],
    batch_size: int = 64,
    max_seq_len: int = 32,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Creates PyTorch Train & Validation DataLoaders for Packet Mode."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_val = int(len(flows) * val_split)
    indices = np.random.permutation(len(flows))
    val_indices, train_indices = indices[:n_val], indices[n_val:]

    train_flows = [flows[i] for i in train_indices]
    val_flows = [flows[i] for i in val_indices]

    train_dataset = PacketSequenceDataset(train_flows, max_seq_len=max_seq_len, fit_scaler=True)
    val_dataset = PacketSequenceDataset(val_flows, max_seq_len=max_seq_len, fit_scaler=False, scaler_stats=train_dataset.scaler_stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "max_seq_len": max_seq_len,
        "feature_dim": 5,
        "scaler_stats": train_dataset.scaler_stats,
        "train_flows_count": len(train_flows),
        "val_flows_count": len(val_flows)
    }

    return train_loader, val_loader, metadata
