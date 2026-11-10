"""
Hybrid Stacking and Blending Ensemble Recommender.
Combines multiple heterogeneous recommendation models:
  - Decoupled Regularized Baseline
  - Item-Item Collaborative Filtering (Residuals + Pearson Shrinkage)
  - Biased Matrix Factorization (MPS GPU)
  - SVD++ with Implicit Feedback (MPS GPU)

Solves for optimal blending weights via constrained Quadratic Programming:
  min_w ||P w - y||_2^2   subject to  sum(w_i) = 1, w_i >= 0
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class HybridEnsemble:
    """
    Constrained Convex Blending Ensemble Recommender.
    Guarantees non-negative weights that sum to 1.0, avoiding over-fitting
    and ensuring the ensemble weakly dominates all single models.
    """

    def __init__(self, model_names: Optional[List[str]] = None, clip_min: float = 1.0, clip_max: float = 5.0):
        self.model_names = model_names or ["Baseline", "Item_kNN", "Biased_MF", "SVDPP"]
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.weights: Optional[np.ndarray] = None
        self.best_rmse: float = float("inf")

    def fit(self, predictions_matrix: np.ndarray, ground_truth: np.ndarray) -> "HybridEnsemble":
        """
        Solves for optimal non-negative weights summing to 1.
        predictions_matrix: shape (N, M) where M is the number of models.
        ground_truth: shape (N,) target ratings.
        """
        n_samples, n_models = predictions_matrix.shape
        assert len(self.model_names) == n_models, f"Expected {len(self.model_names)} models, got {n_models}."

        logger.info(f"Optimizing blending weights for {n_models} models on {n_samples:,} validation samples...")

        # Objective: Mean Squared Error
        def objective(w: np.ndarray) -> float:
            pred = predictions_matrix @ w
            return float(np.mean((ground_truth - pred) ** 2))

        # Initial uniform weights
        w0 = np.full(n_models, 1.0 / n_models, dtype=np.float64)

        # Bounds: 0 <= w_i <= 1
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Constraint: sum(w_i) = 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if not res.success:
            logger.warning(f"Optimization finished with status {res.status}: {res.message}. Using normalized weights.")
            self.weights = np.maximum(0.0, res.x)
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.maximum(0.0, res.x)
            self.weights /= np.sum(self.weights)

        self.best_rmse = float(np.sqrt(np.mean((ground_truth - predictions_matrix @ self.weights) ** 2)))

        weight_dict = {name: round(float(w), 4) for name, w in zip(self.model_names, self.weights)}
        logger.info(f"Optimal Ensemble Weights: {weight_dict}")
        logger.info(f"Ensemble Validation RMSE: {self.best_rmse:.4f}")
        return self

    def predict(self, predictions_matrix: np.ndarray) -> np.ndarray:
        """
        Computes weighted blend of model predictions.
        """
        assert self.weights is not None, "Ensemble must be fitted before predict."
        raw_pred = predictions_matrix @ self.weights
        return np.clip(raw_pred, self.clip_min, self.clip_max)

    def predict_from_models(
        self,
        models: List[Any],
        user_indices: np.ndarray,
        movie_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Generates individual predictions from model list and blends them.
        """
        preds_list = []
        for model in models:
            preds_list.append(model.predict_batch(user_indices, movie_indices))
        P = np.column_stack(preds_list)
        return self.predict(P)

    def get_weights_summary(self) -> Dict[str, float]:
        """Returns dictionary of model names to blending weights."""
        if self.weights is None:
            return {}
        return {name: float(w) for name, w in zip(self.model_names, self.weights)}
