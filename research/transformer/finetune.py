import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from research.transformer.model import PacketTransformerEncoder
from research.data.packet_pipeline import PacketSequenceDataset


def train_or_finetune_transformer(
    train_dataset: PacketSequenceDataset,
    val_dataset: PacketSequenceDataset,
    pretrained_weights_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    seed: int = 42
) -> Tuple[PacketTransformerEncoder, Dict[str, Any]]:
    """
    Fine-tunes or trains the PyTorch Packet Transformer on labeled flow datasets.
    If `pretrained_weights_path` is specified, loads SSL pretrained representations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PacketTransformerEncoder(
        features_dim=5,
        embed_dim=256,
        nhead=4,
        num_layers=4,
        max_seq_len=32,
        num_classes=2
    )

    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        checkpoint = torch.load(pretrained_weights_path, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Load weights non-strictly to allow classification head initialization
        model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        for x, mask, y in train_loader:
            optimizer.zero_grad()
            logits = model(x, padding_mask=mask, return_projection=False)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Evaluation on validation set
    model.eval()
    all_preds = []
    all_probas = []
    all_targets = []

    with torch.no_grad():
        for x, mask, y in val_loader:
            logits = model(x, padding_mask=mask, return_projection=False)
            probas = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probas, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probas.extend(probas[:, 1].cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probas)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) if len(y_true) > 0 else np.zeros((2, 2), dtype=int)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "model_name": "SSL Pretrained Transformer" if pretrained_weights_path else "Supervised Transformer",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    }

    return model, metrics


if __name__ == "__main__":
    from research.data.synthetic_generator import generate_synthetic_packet_sequences
    print("Fine-tuning SSL Pretrained Transformer...")
    flows = generate_synthetic_packet_sequences(n_flows=600, seed=42)
    
    n_val = int(len(flows) * 0.2)
    train_dataset = PacketSequenceDataset(flows[n_val:], fit_scaler=True)
    val_dataset = PacketSequenceDataset(flows[:n_val], fit_scaler=False, scaler_stats=train_dataset.scaler_stats)

    model, metrics = train_or_finetune_transformer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pretrained_weights_path="models/ssl_pretrained_transformer.pt" if os.path.exists("models/ssl_pretrained_transformer.pt") else None,
        epochs=5
    )

    print(f"Fine-Tuning Complete.")
    print(f"Model: {metrics['model_name']}")
    print(f"Accuracy: {metrics['accuracy']} | Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1: {metrics['f1_score']}")

