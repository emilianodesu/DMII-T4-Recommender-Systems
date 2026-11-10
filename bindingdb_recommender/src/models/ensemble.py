"""Gated Hybrid Ensemble Model.
Dynamically detects the recommendation regime for each test pair:
- WARM: Both target and ligand seen during training.
- COLD_LIGAND: Known target, completely novel chemical structure.
- COLD_TARGET: Known ligand, completely novel biological target protein.
- COLD_BOTH: Completely novel target-ligand pair.
Switches and blends prediction weights adaptively across Collaborative, Content,
Biased MF, Two-Tower Neural, and Baseline models.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl

from src.models.baseline import RegularizedBaseline
from src.models.collaborative_knn import CollaborativeResidualKNN
from src.models.content_chemogenomic import ContentChemogenomicKNN
from src.models.latent_factors_gpu import BiasedMatrixFactorization, TwoTowerHybridNeuralMF


class GatedHybridEnsemble:
    """Regime-Aware Gated Hybrid Ensemble for Chemogenomic Recommendation."""

    def __init__(
        self,
        baseline: Optional[RegularizedBaseline] = None,
        collab_knn: Optional[CollaborativeResidualKNN] = None,
        content_knn: Optional[ContentChemogenomicKNN] = None,
        biased_mf: Optional[BiasedMatrixFactorization] = None,
        two_tower: Optional[TwoTowerHybridNeuralMF] = None,
    ):
        self.baseline = baseline or RegularizedBaseline()
        self.collab_knn = collab_knn or CollaborativeResidualKNN(baseline=self.baseline)
        self.content_knn = content_knn or ContentChemogenomicKNN(baseline=self.baseline)
        self.biased_mf = biased_mf or BiasedMatrixFactorization()
        self.two_tower = two_tower or TwoTowerHybridNeuralMF()

        self.known_targets: set = set()
        self.known_ligands: set = set()
        self.is_fitted: bool = False

        # Regime weights: [collab_knn, biased_mf, two_tower, content_knn, baseline]
        self.regime_weights = {
            "WARM": {
                "collab_knn": 0.35,
                "biased_mf": 0.30,
                "two_tower": 0.20,
                "content_knn": 0.15,
                "baseline": 0.00,
            },
            "COLD_LIGAND": {
                "collab_knn": 0.00,
                "biased_mf": 0.00,
                "two_tower": 0.45,
                "content_knn": 0.45,
                "baseline": 0.10,
            },
            "COLD_TARGET": {
                "collab_knn": 0.00,
                "biased_mf": 0.00,
                "two_tower": 0.45,
                "content_knn": 0.45,
                "baseline": 0.10,
            },
            "COLD_BOTH": {
                "collab_knn": 0.00,
                "biased_mf": 0.00,
                "two_tower": 0.70,
                "content_knn": 0.00,
                "baseline": 0.30,
            },
        }

    def fit(self, train_df: pl.DataFrame) -> "GatedHybridEnsemble":
        """Fits all sub-models on training data and indexes known entities."""
        targets = train_df["target_id"].to_list()
        ligands = train_df["ligand_smiles"].to_list()

        self.known_targets = set(targets)
        self.known_ligands = set(ligands)

        print("[Ensemble] Fitting Baseline...")
        self.baseline.fit(train_df)

        print("[Ensemble] Fitting Collaborative Residual k-NN...")
        self.collab_knn.fit(train_df)

        print("[Ensemble] Fitting Content Chemogenomic k-NN...")
        self.content_knn.fit(train_df)

        print("[Ensemble] Fitting GPU Biased Matrix Factorization...")
        self.biased_mf.fit(train_df)

        print("[Ensemble] Fitting GPU Two-Tower Hybrid Neural MF...")
        self.two_tower.fit(train_df)

        self.is_fitted = True
        return self

    def get_regime(self, target_id: str, ligand_smiles: str) -> str:
        """Determines the regime for a pair."""
        in_tgt = target_id in self.known_targets
        in_lig = ligand_smiles in self.known_ligands

        if in_tgt and in_lig:
            return "WARM"
        elif in_tgt and not in_lig:
            return "COLD_LIGAND"
        elif not in_tgt and in_lig:
            return "COLD_TARGET"
        else:
            return "COLD_BOTH"

    def predict_batch(
        self,
        targets: Union[List[str], pl.Series],
        ligands: Union[List[str], pl.Series],
        sequences: Optional[Union[List[str], pl.Series]] = None,
    ) -> np.ndarray:
        """Computes gated predictions combining models according to pair regime."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before calling predict.")
        if isinstance(targets, pl.Series):
            targets = targets.to_list()
        if isinstance(ligands, pl.Series):
            ligands = ligands.to_list()
        seq_list = sequences.to_list() if isinstance(sequences, pl.Series) else sequences

        n = len(targets)
        # 1. Obtain predictions from each component
        pred_base = self.baseline.predict_batch(targets, ligands)
        pred_cknn = self.collab_knn.predict_batch(targets, ligands)
        pred_cont = self.content_knn.predict_batch(targets, ligands, sequences=seq_list)
        pred_bmf = self.biased_mf.predict_batch(targets, ligands)
        pred_twot = self.two_tower.predict_batch(targets, ligands, sequences=seq_list)

        ensemble_preds = np.empty(n, dtype=np.float32)

        for i in range(n):
            t = targets[i]
            l = ligands[i]
            regime = self.get_regime(t, l)
            w = self.regime_weights[regime]

            p = (
                w["collab_knn"] * pred_cknn[i]
                + w["biased_mf"] * pred_bmf[i]
                + w["two_tower"] * pred_twot[i]
                + w["content_knn"] * pred_cont[i]
                + w["baseline"] * pred_base[i]
            )
            ensemble_preds[i] = p

        return ensemble_preds
