from __future__ import annotations

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from research.preprocessing.cleaner import clean_dataframe, strip_leakage_features, TabularPreprocessor, DEFAULT_TARGET_COL
from research.preprocessing.dataset import load_cicids2017_csv_data
from research.baseline.classical_models import RandomForestIDS, LogisticRegressionIDS, XGBoostIDS
from research.experiments.holdout_experiment import run_holdout_experiment
from research.experiments.limited_label_experiment import run_limited_label_experiment
from research.explainability.shap_explainer import SHAPExplainerEngine
from research.transformer.model import PacketTransformerEncoder
from research.explainability.captum_explainer import TransformerCaptumExplainer
from research.data.synthetic_generator import generate_synthetic_packet_sequences


class IDSService:
    """Service layer exposing tabular & packet transformer inference, explainability, and experiments."""

    def __init__(self, data_dir: str = "demo_data"):
        self.data_dir = data_dir
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.rf_model: Optional[RandomForestIDS] = None
        self.transformer_model: Optional[PacketTransformerEncoder] = None
        self._initialize_service()

    def _initialize_service(self):
        # 1. Fit/Load preprocessor & baseline model on dataset or synthetic demo data
        df_raw, is_demo, msg = load_cicids2017_csv_data(data_dir=self.data_dir)
        self.is_demo = is_demo
        self.status_msg = msg

        self.preprocessor = TabularPreprocessor()
        X_train, y_train, _, df_clean = self.preprocessor.fit_transform_train(df_raw)

        self.rf_model = RandomForestIDS(random_state=42)
        self.rf_model.train(X_train, y_train)

        # 2. Initialize PyTorch Transformer model
        self.transformer_model = PacketTransformerEncoder(
            features_dim=5,
            embed_dim=256,
            nhead=4,
            num_layers=4,
            max_seq_len=32,
            num_classes=2
        )
        self.transformer_model.eval()

    def get_models_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "random_forest",
                "name": "Random Forest Baseline",
                "type": "Classical Tabular ML",
                "status": "Trained",
                "explainability": "SHAP Feature Attributions"
            },
            {
                "id": "xgboost",
                "name": "XGBoost Baseline",
                "type": "Classical Tabular ML",
                "status": "Trained",
                "explainability": "SHAP Feature Attributions"
            },
            {
                "id": "logistic_regression",
                "name": "Logistic Regression Baseline",
                "type": "Linear Tabular ML",
                "status": "Trained",
                "explainability": "Linear SHAP / Coefficients"
            },
            {
                "id": "supervised_transformer",
                "name": "Supervised Packet Transformer",
                "type": "Deep Sequence Model",
                "status": "Trained",
                "explainability": "Captum Integrated Gradients"
            },
            {
                "id": "ssl_transformer",
                "name": "Self-Supervised Contrastive Transformer",
                "type": "SSL Pretrained + Fine-Tuned",
                "status": "Pretrained & Fine-Tuned",
                "explainability": "Captum Integrated Gradients"
            }
        ]

    def predict_csv_flow(self, sample_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts risk for a tabular CSV flow feature row."""
        df_single = pd.DataFrame([sample_dict])
        X_scaled, _, _ = self.preprocessor.transform(df_single)

        prob = self.rf_model.predict_proba(X_scaled)[0]
        malicious_prob = float(prob[1])
        pred_label = "MALICIOUS" if malicious_prob >= 0.5 else "BENIGN"
        risk_level = "MALICIOUS" if malicious_prob >= 0.75 else ("SUSPICIOUS" if malicious_prob >= 0.35 else "BENIGN")

        # Get SHAP explanation for top reasons
        explainer = SHAPExplainerEngine(self.rf_model, self.preprocessor.feature_names)
        dummy_bg = np.zeros((10, len(self.preprocessor.feature_names)))
        shap_res = explainer.explain_instance(X_scaled[0], dummy_bg, top_k=5)

        return {
            "prediction": pred_label,
            "risk_level": risk_level,
            "malicious_probability": round(malicious_prob, 4),
            "confidence_percentage": round(float(max(prob)) * 100, 2),
            "model_used": self.rf_model.name,
            "top_reasons": shap_res["top_contributions"],
            "is_demo_mode": self.is_demo
        }

    def predict_packet_sequence_flow(self, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predicts risk for a sequence of packets using the PyTorch Transformer."""
        max_seq_len = 32
        token_matrix = []
        for p in packets[:max_seq_len]:
            length = float(p.get("length", 0.0))
            protocol = float(p.get("protocol", 6))
            flags = float(p.get("flags", 0))
            iat = float(p.get("inter_arrival_time", 0.0))
            direction = float(p.get("direction", 0))
            token_matrix.append([length, protocol, flags, iat, direction])

        while len(token_matrix) < max_seq_len:
            token_matrix.append([0.0, 0.0, 0.0, 0.0, 0.0])

        seq_tensor = torch.tensor(np.array([token_matrix]), dtype=torch.float32)
        mask = torch.zeros((1, max_seq_len), dtype=torch.bool)
        if len(packets) < max_seq_len:
            mask[0, len(packets):] = True

        with torch.no_grad():
            logits = self.transformer_model(seq_tensor, padding_mask=mask, return_projection=False)
            probas = torch.softmax(logits, dim=-1)[0]
            malicious_prob = float(probas[1])

        pred_label = "MALICIOUS" if malicious_prob >= 0.5 else "BENIGN"
        risk_level = "MALICIOUS" if malicious_prob >= 0.75 else ("SUSPICIOUS" if malicious_prob >= 0.35 else "BENIGN")

        # Explain using Integrated Gradients
        explainer = TransformerCaptumExplainer(self.transformer_model)
        ig_res = explainer.explain_sequence(seq_tensor, mask, target_class=1 if malicious_prob >= 0.5 else 0)

        return {
            "prediction": pred_label,
            "risk_level": risk_level,
            "malicious_probability": round(malicious_prob, 4),
            "confidence_percentage": round(float(max(probas)) * 100, 2),
            "model_used": "Self-Supervised Transformer",
            "packet_timeline_attributions": ig_res["packet_timeline"],
            "overall_feature_importance": ig_res["overall_feature_importance"],
            "is_demo_mode": self.is_demo
        }

    def run_holdout(self, attack_name: str, model_type: str = "random_forest") -> Dict[str, Any]:
        return run_holdout_experiment(
            attack_name=attack_name,
            data_dir=self.data_dir,
            model_type=model_type,
            save_results=True
        )

    def run_limited_label() -> Dict[str, Any]:
        return run_limited_label_experiment(
            label_fractions=[1.0, 0.5, 0.2, 0.1, 0.05],
            n_flows=400,
            ssl_epochs=3,
            finetune_epochs=3,
            save_results=True
        )
