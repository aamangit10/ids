from __future__ import annotations

import os
import json
import torch
import numpy as np
from typing import Dict, Any, List
from research.data.synthetic_generator import generate_synthetic_packet_sequences
from research.data.packet_pipeline import PacketSequenceDataset
from research.ssl.pretrain import pretrain_ssl_transformer
from research.transformer.finetune import train_or_finetune_transformer


def run_limited_label_experiment(
    label_fractions: List[float] = [1.0, 0.5, 0.2, 0.1, 0.05],
    n_flows: int = 600,
    ssl_epochs: int = 5,
    finetune_epochs: int = 5,
    save_results: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates data scarcity efficiency comparing:
    Supervised Transformer vs Self-Supervised Pretrained + Fine-Tuned Transformer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    flows = generate_synthetic_packet_sequences(n_flows=n_flows, seed=seed)

    # 1. SSL Pretraining on all unlabeled flows
    ssl_model_path = "models/temp_ssl_pretrained.pt"
    pretrain_ssl_transformer(flows, epochs=ssl_epochs, save_path=ssl_model_path, seed=seed)

    # Split dataset into train (80%) and val (20%)
    n_val = int(len(flows) * 0.2)
    indices = np.random.permutation(len(flows))
    val_flows = [flows[i] for i in indices[:n_val]]
    full_train_flows = [flows[i] for i in indices[n_val:]]

    val_dataset = PacketSequenceDataset(val_flows, fit_scaler=False)

    results = []

    for frac in label_fractions:
        n_sub = max(10, int(len(full_train_flows) * frac))
        sub_train_flows = full_train_flows[:n_sub]

        train_dataset = PacketSequenceDataset(sub_train_flows, fit_scaler=True)

        # Train Supervised Transformer (from scratch)
        _, sup_metrics = train_or_finetune_transformer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            pretrained_weights_path=None,
            epochs=finetune_epochs,
            seed=seed
        )

        # Fine-tune SSL Pretrained Transformer
        _, ssl_metrics = train_or_finetune_transformer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            pretrained_weights_path=ssl_model_path,
            epochs=finetune_epochs,
            seed=seed
        )

        results.append({
            "label_fraction": frac,
            "label_percentage_str": f"{int(frac * 100)}%",
            "num_labeled_samples": n_sub,
            "supervised_transformer": sup_metrics,
            "ssl_pretrained_transformer": ssl_metrics
        })

    experiment_data = {
        "title": "Limited-Label Efficiency Experiment",
        "description": "Comparative evaluation of Supervised vs SSL Pretrained Transformer across limited labeled dataset fractions.",
        "results": results
    }

    if save_results:
        os.makedirs("results", exist_ok=True)
        with open("results/limited_label_results.json", "w") as f:
            json.dump(experiment_data, f, indent=2)

    return experiment_data
