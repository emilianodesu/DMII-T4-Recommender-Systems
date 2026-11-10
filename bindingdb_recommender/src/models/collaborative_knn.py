"""Collaborative Filtering k-NN Model on Centered Residuals.
Implements Target-based and Ligand-based collaborative nearest neighbors on residuals:
e_tl = y_tl - b_tl
y_hat_tl = b_tl + sum(s * e) / (sum|s| + beta)
where beta shrinkage contracts towards baseline b_tl (never towards 0).
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.preprocessing import normalize

from src.config import CollaborativeKNNConfig
from src.models.baseline import RegularizedBaseline


class CollaborativeResidualKNN:
    """Collaborative k-NN Recommender operating on baseline-centered residuals."""

    def __init__(
        self,
        config: Optional[CollaborativeKNNConfig] = None,
        baseline: Optional[RegularizedBaseline] = None,
    ):
        self.config = config or CollaborativeKNNConfig()
        self.baseline = baseline or RegularizedBaseline()

        self.target_to_idx: Dict[str, int] = {}
        self.idx_to_target: Dict[int, str] = {}
        self.ligand_to_idx: Dict[str, int] = {}
        self.idx_to_ligand: Dict[int, str] = {}

        self.residual_matrix_csr: Optional[sparse.csr_matrix] = None
        self.residual_matrix_csc: Optional[sparse.csc_matrix] = None
        self.target_sim_matrix: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, train_df: pl.DataFrame) -> "CollaborativeResidualKNN":
        """Fits baseline, computes residuals, builds sparse interaction matrix,
        and computes normalized target similarity matrix.
        """
        # 1. Fit regularized baseline
        self.baseline.fit(train_df)

        targets = train_df["target_id"].to_list()
        ligands = train_df["ligand_smiles"].to_list()
        pIC50 = train_df["pIC50"].to_numpy().astype(np.float64)

        # 2. Build index mappings
        unique_targets = sorted(list(set(targets)))
        unique_ligands = sorted(list(set(ligands)))

        self.target_to_idx = {t: i for i, t in enumerate(unique_targets)}
        self.idx_to_target = {i: t for i, t in enumerate(unique_targets)}
        self.ligand_to_idx = {l: j for j, l in enumerate(unique_ligands)}
        self.idx_to_ligand = {j: l for j, l in enumerate(unique_ligands)}

        n_targets = len(unique_targets)
        n_ligands = len(unique_ligands)

        row_indices = np.array([self.target_to_idx[t] for t in targets], dtype=np.int32)
        col_indices = np.array([self.ligand_to_idx[l] for l in ligands], dtype=np.int32)

        # 3. Compute centered residuals: e_tl = y_tl - (mu + b_t + b_l)
        baseline_preds = self.baseline.predict_batch(targets, ligands)
        residuals = pIC50 - baseline_preds

        # 4. Create sparse matrices
        self.residual_matrix_csr = sparse.csr_matrix(
            (residuals, (row_indices, col_indices)),
            shape=(n_targets, n_ligands),
            dtype=np.float32,
        )
        self.residual_matrix_csc = self.residual_matrix_csr.tocsc()

        # 5. Compute Target-Target cosine similarity on normalized residual vectors
        # Normalized residual row vectors
        norm_csr = normalize(self.residual_matrix_csr, norm="l2", axis=1)
        sim = (norm_csr @ norm_csr.T).toarray().astype(np.float32)

        # Remove self-similarity on the diagonal
        np.fill_diagonal(sim, 0.0)
        self.target_sim_matrix = sim

        self.is_fitted = True
        return self

    def predict_pair(self, target_id: str, ligand_smiles: str) -> float:
        """Predicts affinity pIC50 with shrinkage towards b_tl."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        # Always start from baseline prior
        b_tl = self.baseline.predict_pair(target_id, ligand_smiles)

        # Check if target or ligand is cold
        if (
            target_id not in self.target_to_idx
            or ligand_smiles not in self.ligand_to_idx
            or self.residual_matrix_csc is None
            or self.target_sim_matrix is None
        ):
            return b_tl

        t_idx = self.target_to_idx[target_id]
        l_idx = self.ligand_to_idx[ligand_smiles]

        # Extract all targets that have tested this ligand
        col_start = self.residual_matrix_csc.indptr[l_idx]
        col_end = self.residual_matrix_csc.indptr[l_idx + 1]

        if col_start == col_end:
            # Ligand has no other targets in train
            return b_tl

        rated_targets = self.residual_matrix_csc.indices[col_start:col_end]
        col_residuals = self.residual_matrix_csc.data[col_start:col_end]

        # Exclude self if target was in rated_targets
        valid_mask = rated_targets != t_idx
        if not np.any(valid_mask):
            return b_tl

        candidate_targets = rated_targets[valid_mask]
        candidate_residuals = col_residuals[valid_mask]
        sims = self.target_sim_matrix[t_idx, candidate_targets]

        # Keep positive similarities
        pos_mask = sims > 0.0
        if not np.any(pos_mask):
            return b_tl

        pos_sims = sims[pos_mask]
        pos_residuals = candidate_residuals[pos_mask]

        # Top-k selection
        k = self.config.k
        if len(pos_sims) > k:
            top_indices = np.argpartition(pos_sims, -k)[-k:]
            pos_sims = pos_sims[top_indices]
            pos_residuals = pos_residuals[top_indices]

        # Regularized residual adjustment with shrinkage beta:
        # delta = sum(s * e) / (sum(|s|) + beta)
        num = np.sum(pos_sims * pos_residuals)
        denom = np.sum(pos_sims) + self.config.shrinkage_beta
        delta = num / denom

        return float(b_tl + delta)

    def predict_batch(
        self,
        targets: Union[List[str], pl.Series],
        ligands: Union[List[str], pl.Series],
    ) -> np.ndarray:
        """Batch prediction iterating through pairs."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        if isinstance(targets, pl.Series):
            targets = targets.to_list()
        if isinstance(ligands, pl.Series):
            ligands = ligands.to_list()

        preds = np.empty(len(targets), dtype=np.float32)
        for i, (t, l) in enumerate(zip(targets, ligands)):
            preds[i] = self.predict_pair(t, l)
        return preds
