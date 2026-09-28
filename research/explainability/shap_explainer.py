import os
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from research.baseline.base_model import BaseIDSModel


class SHAPExplainerEngine:
    """
    Computes local and global feature importance attributions for tabular classical models using SHAP.
    """

    def __init__(self, model: BaseIDSModel, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[Any] = None

    def _init_explainer(self, background_data: np.ndarray):
        """Initializes appropriate SHAP explainer based on model family."""
        if self.explainer is not None:
            return

        raw_model = self.model.model
        model_name = self.model.name.lower()

        # Choose background sample size for efficiency
        bg_sample = background_data[:100] if len(background_data) > 100 else background_data

        try:
            if "random forest" in model_name or "xgboost" in model_name or "tree" in model_name:
                self.explainer = shap.TreeExplainer(raw_model)
            elif "logistic" in model_name or "linear" in model_name:
                self.explainer = shap.LinearExplainer(raw_model, bg_sample)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
        except Exception as e:
            # Fallback to Kernel/Explainer
            self.explainer = shap.Explainer(self.model.predict_proba, bg_sample)

    def explain_instance(self, sample_x: np.ndarray, background_data: np.ndarray, top_k: int = 8) -> Dict[str, Any]:
        """
        Explains an individual network flow sample's prediction.
        Returns top-k positive and negative feature contributions.
        """
        if sample_x.ndim == 1:
            sample_x = sample_x.reshape(1, -1)

        self._init_explainer(background_data)

        # Predict label and confidence
        prob = self.model.predict_proba(sample_x)[0]
        pred_class = int(np.argmax(prob))
        malicious_prob = float(prob[1])

        # Compute SHAP values
        shap_vals = self.explainer(sample_x)
        
        # Handle SHAP output dimensions (binary / multiclass / 2D matrix)
        values = shap_vals.values[0]
        if values.ndim == 2:  # Shape: (features, classes) -> take class 1 (ATTACK)
            values = values[:, 1]
        elif values.ndim == 1:
            values = values

        # Pair feature names with SHAP values & original feature values
        contributions = []
        for idx, val in enumerate(values):
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            feat_val = float(sample_x[0, idx])
            contributions.append({
                "feature": feat_name,
                "feature_value": round(feat_val, 4),
                "shap_value": round(float(val), 4),
                "abs_shap_value": round(float(abs(val)), 4)
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: x["abs_shap_value"], reverse=True)

        top_contributions = contributions[:top_k]

        return {
            "prediction": "MALICIOUS" if pred_class == 1 else "BENIGN",
            "prediction_class": pred_class,
            "malicious_probability": round(malicious_prob, 4),
            "confidence_percentage": round(max(prob) * 100, 2),
            "top_contributions": top_contributions,
            "explanation_summary": f"Top indicator: '{top_contributions[0]['feature']}' (SHAP: {top_contributions[0]['shap_value']:+.4f})" if top_contributions else "No features available"
        }

    def explain_global(self, dataset_x: np.ndarray, top_k: int = 15) -> Dict[str, Any]:
        """Computes global feature importance across a dataset slice."""
        self._init_explainer(dataset_x)
        sample_subset = dataset_x[:200] if len(dataset_x) > 200 else dataset_x
        
        shap_vals = self.explainer(sample_subset)
        values = shap_vals.values
        if values.ndim == 3:  # (samples, features, classes) -> class 1
            values = values[:, :, 1]

        mean_abs = np.mean(np.abs(values), axis=0)
        
        global_importance = []
        for idx, score in enumerate(mean_abs):
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            global_importance.append({
                "feature": feat_name,
                "importance_score": round(float(score), 4)
            })

        global_importance.sort(key=lambda x: x["importance_score"], reverse=True)

        return {
            "global_top_features": global_importance[:top_k]
        }
