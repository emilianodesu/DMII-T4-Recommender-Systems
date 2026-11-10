import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Unit and Integration Tests for BindingDB Chemogenomic Recommender System.
Verifies data curation, scaffold-split zero-leakage, target-split zero-leakage,
mathematical properties of baseline, residual k-NN, Tanimoto content k-NN,
GPU Latent Factor models (MPS), and recommender ranking metrics.
Uses standard library unittest for seamless zero-dependency execution.
"""
import unittest
import numpy as np
import polars as pl
import torch

from src.data_pipeline import (
    extract_canonical_and_scaffold,
    split_random,
    split_scaffold,
    split_target,
)
from src.featurizer import MoleculeFeaturizer, ProteinFeaturizer
from src.models.baseline import RegularizedBaseline
from src.models.collaborative_knn import CollaborativeResidualKNN
from src.models.content_chemogenomic import ContentChemogenomicKNN, compute_tanimoto_vectorized
from src.models.latent_factors_gpu import BiasedMatrixFactorization, TwoTowerHybridNeuralMF
from src.models.ensemble import GatedHybridEnsemble
from src.evaluation import compute_regression_metrics, compute_ranking_metrics_for_target


def get_sample_dataset() -> pl.DataFrame:
    """Generates a small controlled chemogenomic DataFrame with known ground truth."""
    data = {
        "target_id": [
            "T1", "T1", "T1", "T1",
            "T2", "T2", "T2", "T2",
            "T3", "T3", "T3",
        ],
        "target_name": [
            "Kinase A", "Kinase A", "Kinase A", "Kinase A",
            "Kinase B", "Kinase B", "Kinase B", "Kinase B",
            "Protease C", "Protease C", "Protease C",
        ],
        "ligand_smiles": [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
            "c1ccccc1",  # Benzene
            "c1ccncc1",  # Pyridine
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "c1ccccc1",
            "CCO",  # Ethanol
            "CC(=O)Oc1ccccc1C(=O)O",
            "c1ccncc1",
            "CCO",
        ],
        "sequence": [
            "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSS",
            "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSS",
            "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSS",
            "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSS",
            "MVKVYWSKTLGLKGSFLLDTVGGSFLGSRKLVEETSKKKQLEKFLVDLPVGKLVLLDSRKLEKKGGF",
            "MVKVYWSKTLGLKGSFLLDTVGGSFLGSRKLVEETSKKKQLEKFLVDLPVGKLVLLDSRKLEKKGGF",
            "MVKVYWSKTLGLKGSFLLDTVGGSFLGSRKLVEETSKKKQLEKFLVDLPVGKLVLLDSRKLEKKGGF",
            "MVKVYWSKTLGLKGSFLLDTVGGSFLGSRKLVEETSKKKQLEKFLVDLPVGKLVLLDSRKLEKKGGF",
            "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASAD",
            "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASAD",
            "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASAD",
        ],
        "pIC50": [
            7.5, 6.2, 5.0, 4.8,
            8.0, 6.5, 5.2, 4.5,
            7.1, 5.5, 4.0,
        ],
        "scaffold": [
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccncc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "acyclic_CCO",
            "c1ccccc1",
            "c1ccncc1",
            "acyclic_CCO",
        ],
    }
    return pl.DataFrame(data)


class TestBindingDBRecSys(unittest.TestCase):

    def setUp(self):
        self.sample_df = get_sample_dataset()

    def test_rdkit_canonical_and_scaffold(self):
        s = "CC(=O)Oc1ccccc1C(=O)O"
        can, scaf = extract_canonical_and_scaffold(s)
        self.assertIsNotNone(can)
        self.assertEqual(scaf, "c1ccccc1")

        # Invalid SMILES
        bad_can, bad_scaf = extract_canonical_and_scaffold("INVALID_XYZ_123")
        self.assertIsNone(bad_can)
        self.assertIsNone(bad_scaf)

    def test_scaffold_split_zero_leakage(self):
        """Verifies that Bemis-Murcko scaffold split has ZERO scaffold overlap between train and test."""
        splits = split_scaffold(self.sample_df, train_ratio=0.6, val_ratio=0.2, seed=42)
        train_scafs = set(splits["train"]["scaffold"].unique().to_list())
        test_scafs = set(splits["test"]["scaffold"].unique().to_list())

        intersection = train_scafs.intersection(test_scafs)
        self.assertEqual(len(intersection), 0, f"Scaffold leakage detected: {intersection}")

    def test_target_split_zero_leakage(self):
        """Verifies that Target split has ZERO target overlap between train and test."""
        splits = split_target(self.sample_df, train_ratio=0.6, val_ratio=0.2, seed=42)
        train_targets = set(splits["train"]["target_id"].unique().to_list())
        test_targets = set(splits["test"]["target_id"].unique().to_list())

        intersection = train_targets.intersection(test_targets)
        self.assertEqual(len(intersection), 0, f"Target leakage detected: {intersection}")

    def test_regularized_baseline(self):
        """Verifies that Baseline correctly calculates global mean and regularized biases."""
        baseline = RegularizedBaseline()
        baseline.fit(self.sample_df)

        self.assertTrue(baseline.is_fitted)
        self.assertTrue(5.0 < baseline.global_mean < 7.0)

        # Test prediction on known pair
        pred = baseline.predict_pair("T1", "CC(=O)Oc1ccccc1C(=O)O")
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 6.0)

        # Test fallback on unknown pair
        pred_unknown = baseline.predict_pair("UNKNOWN_T", "UNKNOWN_L")
        self.assertTrue(np.isclose(pred_unknown, baseline.global_mean))

    def test_collaborative_knn_shrinkage(self):
        """Verifies that collaborative residual k-NN contracts towards b_tl."""
        collab = CollaborativeResidualKNN()
        collab.fit(self.sample_df)

        # For unknown entity, prediction must equal baseline b_tl exactly
        b_tl = collab.baseline.predict_pair("UNKNOWN_TARGET", "CCO")
        pred_knn = collab.predict_pair("UNKNOWN_TARGET", "CCO")
        self.assertTrue(np.isclose(b_tl, pred_knn))

    def test_content_chemogenomic_tanimoto(self):
        """Verifies vectorized Tanimoto similarity math."""
        q = np.array([1, 1, 0, 0], dtype=np.float32)
        db = np.array([
            [1, 1, 0, 0],  # Identical -> T = 1.0
            [1, 0, 0, 0],  # 1 common, 2 union -> T = 0.5
            [0, 0, 1, 1],  # 0 common -> T = 0.0
        ], dtype=np.float32)

        sims = compute_tanimoto_vectorized(q, db)
        self.assertTrue(np.isclose(sims[0], 1.0))
        self.assertTrue(np.isclose(sims[1], 0.5))
        self.assertTrue(np.isclose(sims[2], 0.0))

    def test_latent_factors_gpu_mps(self):
        """Verifies Biased Matrix Factorization and Two-Tower Hybrid run smoothly."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # 1. Biased MF
        bmf = BiasedMatrixFactorization()
        bmf.config.device = device
        bmf.config.epochs = 5
        bmf.config.batch_size = 4
        bmf.fit(self.sample_df)
        self.assertTrue(bmf.is_fitted)

        preds = bmf.predict_batch(["T1", "UNKNOWN_T"], ["CCO", "UNKNOWN_L"])
        self.assertEqual(len(preds), 2)
        self.assertFalse(np.isnan(preds).any())

        # 2. Two-Tower Hybrid
        tt = TwoTowerHybridNeuralMF()
        tt.config.device = device
        tt.config.epochs = 5
        tt.config.batch_size = 4
        tt.fit(self.sample_df)
        self.assertTrue(tt.is_fitted)

        preds_tt = tt.predict_batch(
            ["T1", "UNKNOWN_T"],
            ["CCO", "UNKNOWN_L"],
            sequences=["MGSNKSKPKDASQ", "MVKVYWSKTLGL"],
        )
        self.assertEqual(len(preds_tt), 2)
        self.assertFalse(np.isnan(preds_tt).any())

    def test_evaluation_ranking_metrics(self):
        """Verifies NDCG@K and Recall@K calculation on known vectors."""
        y_true = np.array([7.5, 6.5, 5.0, 4.0])  # Actives at indices 0 and 1
        y_pred_perfect = np.array([8.0, 7.0, 5.0, 3.0])  # Perfect ranking
        y_pred_reverse = np.array([3.0, 4.0, 7.0, 8.0])  # Worst ranking

        ndcg_p, rec_p = compute_ranking_metrics_for_target(y_true, y_pred_perfect, k=2, threshold=6.0)
        self.assertTrue(np.isclose(ndcg_p, 1.0))
        self.assertTrue(np.isclose(rec_p, 1.0))

        ndcg_r, rec_r = compute_ranking_metrics_for_target(y_true, y_pred_reverse, k=2, threshold=6.0)
        self.assertLess(ndcg_r, 0.1)
        self.assertTrue(np.isclose(rec_r, 0.0))


if __name__ == "__main__":
    unittest.main()
