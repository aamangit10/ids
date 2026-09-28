import os
import joblib
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


class BaseIDSModel(ABC):
    """
    Abstract Base Class establishing a uniform API across all classical & deep learning IDS models.
    """

    def __init__(self, name: str):
        self.name = name
        self.model: Any = None
        self.is_trained: bool = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Fit the model on feature matrix X and target y."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0=BENIGN, 1=ATTACK)."""
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"Model '{self.name}' has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"Model '{self.name}' has not been trained yet.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            # Sigmoid normalization
            proba_1 = 1.0 / (1.0 + np.exp(-df))
            return np.column_stack([1.0 - proba_1, proba_1])
        else:
            preds = self.predict(X)
            return np.column_stack([1.0 - preds, preds])

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Calculates comprehensive IDS security evaluation metrics:
        Accuracy, Precision, Recall, F1, ROC-AUC, False Positive Rate (FPR), Confusion Matrix.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if X.shape[0] > 0 else np.array([])

        acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
        prec = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0
        rec = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0

        try:
            auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5

        # Confusion Matrix breakdown
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) if len(y_true) > 0 else np.zeros((2, 2), dtype=int)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # False Positive Rate = FP / (FP + TN)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "model_name": self.name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
            }
        }

    def save(self, filepath: str) -> None:
        """Serialize trained model state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({"name": self.name, "model": self.model, "is_trained": self.is_trained}, filepath)

    def load(self, filepath: str) -> None:
        """Load serialized model state from disk."""
        data = joblib.load(filepath)
        self.name = data.get("name", self.name)
        self.model = data.get("model")
        self.is_trained = data.get("is_trained", False)
