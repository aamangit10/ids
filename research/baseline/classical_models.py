import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from research.baseline.base_model import BaseIDSModel

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class LogisticRegressionIDS(BaseIDSModel):
    """Logistic Regression baseline for fast linear tabular classification."""

    def __init__(self, max_iter: int = 1000, solver: str = "lbfgs", random_state: int = 42):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=max_iter, solver=solver, random_state=random_state)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.model.fit(X, y)
        self.is_trained = True
        return self.evaluate(X, y)


class RandomForestIDS(BaseIDSModel):
    """Random Forest baseline for non-linear decision boundary classification."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 15, random_state: int = 42, n_jobs: int = -1):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.model.fit(X, y)
        self.is_trained = True
        return self.evaluate(X, y)


class XGBoostIDS(BaseIDSModel):
    """XGBoost baseline (with fallback to Scikit-Learn GradientBoosting if XGBoost is absent)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6, random_state: int = 42):
        super().__init__("XGBoost")
        if XGB_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                eval_metric="logloss"
            )
        else:
            self.name = "Gradient Boosting (XGBoost Fallback)"
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.model.fit(X, y)
        self.is_trained = True
        return self.evaluate(X, y)
