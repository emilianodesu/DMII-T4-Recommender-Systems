"""Regularized Biochemical Baseline Model.
Implements the foundational additive recommender model:
y_hat(t, l) = mu + b_target + b_ligand
with L2 shrinkage towards global mean to prevent overfitting on sparsely assayed pairs.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import polars as pl

from src.config import BaselineConfig


class RegularizedBaseline:
    """Biochemical Baseline with L2 regularized target and ligand biases."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.global_mean: float = 0.0
        self.b_target: Dict[str, float] = {}
        self.b_ligand: Dict[str, float] = {}
        self.target_counts: Dict[str, int] = {}
        self.ligand_counts: Dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: Union[pl.DataFrame, Dict[str, np.ndarray]]) -> "RegularizedBaseline":
        """Fits the regularized baseline on training data."""
        if isinstance(train_df, pl.DataFrame):
            targets = train_df["target_id"].to_list()
            ligands = train_df["ligand_smiles"].to_list()
            pIC50 = train_df["pIC50"].to_numpy().astype(np.float64)
        else:
            targets = train_df["target_id"]
            ligands = train_df["ligand_smiles"]
            pIC50 = np.asarray(train_df["pIC50"], dtype=np.float64)

        n_samples = len(pIC50)
        if n_samples == 0:
            raise ValueError("Cannot fit RegularizedBaseline on empty dataset.")

        # 1. Global biochemical mean pIC50
        self.global_mean = float(np.mean(pIC50))

        # 2. Accumulate target residuals and counts
        target_residuals: Dict[str, float] = {}
        self.target_counts = {}
        for t, y in zip(targets, pIC50):
            res = y - self.global_mean
            target_residuals[t] = target_residuals.get(t, 0.0) + res
            self.target_counts[t] = self.target_counts.get(t, 0) + 1

        # Regularized target bias: b_t = sum(y_tl - mu) / (|R(t)| + lambda_t)
        self.b_target = {
            t: target_residuals[t] / (self.target_counts[t] + self.config.lambda_target)
            for t in target_residuals
        }

        # 3. Accumulate ligand residuals: y_tl - mu - b_t
        ligand_residuals: Dict[str, float] = {}
        self.ligand_counts = {}
        for t, l, y in zip(targets, ligands, pIC50):
            bt = self.b_target.get(t, 0.0)
            res = y - self.global_mean - bt
            ligand_residuals[l] = ligand_residuals.get(l, 0.0) + res
            self.ligand_counts[l] = self.ligand_counts.get(l, 0) + 1

        # Regularized ligand bias: b_l = sum(y_tl - mu - b_t) / (|R(l)| + lambda_l)
        self.b_ligand = {
            l: ligand_residuals[l] / (self.ligand_counts[l] + self.config.lambda_ligand)
            for l in ligand_residuals
        }

        self.is_fitted = True
        return self

    def predict_pair(self, target_id: str, ligand_smiles: str) -> float:
        """Predicts pIC50 for a single (target, ligand) pair."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        bt = self.b_target.get(target_id, 0.0)
        bl = self.b_ligand.get(ligand_smiles, 0.0)
        return float(self.global_mean + bt + bl)

    def predict_batch(
        self,
        targets: Union[List[str], pl.Series],
        ligands: Union[List[str], pl.Series],
    ) -> np.ndarray:
        """Vectorized prediction for arrays of target IDs and ligand SMILES."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        if isinstance(targets, pl.Series):
            targets = targets.to_list()
        if isinstance(ligands, pl.Series):
            ligands = ligands.to_list()

        mu = self.global_mean
        b_tgt_map = self.b_target
        b_lig_map = self.b_ligand

        preds = np.empty(len(targets), dtype=np.float32)
        for i, (t, l) in enumerate(zip(targets, ligands)):
            preds[i] = mu + b_tgt_map.get(t, 0.0) + b_lig_map.get(l, 0.0)

        return preds

    def get_residuals(
        self,
        targets: Union[List[str], pl.Series],
        ligands: Union[List[str], pl.Series],
        y_true: Union[np.ndarray, pl.Series],
    ) -> np.ndarray:
        """Computes residuals: e_tl = y_true - y_hat."""
        preds = self.predict_batch(targets, ligands)
        if isinstance(y_true, pl.Series):
            y_arr = y_true.to_numpy().astype(np.float32)
        else:
            y_arr = np.asarray(y_true, dtype=np.float32)
        return y_arr - preds
