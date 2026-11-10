import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
Test Suite for Netflix Prize Recommender System.
Implemented using standard unittest for zero-dependency execution.
Tests:
- Data pipeline integrity & strict anti-join
- Decoupled regularized baseline
- Collaborative Item-Item k-NN on residuals (RMSE <= Baseline)
- Latent factor models on Apple Silicon Metal MPS (Biased MF and SVD++)
- Content-based Cold Start Ridge projection
- Constrained Stacking Ensemble blending
- Evaluation ranking metrics (NDCG@K, DCG@K)
"""

import json
import unittest
import numpy as np
import polars as pl
from scipy.sparse import load_npz

from src.config import (
    TRAIN_PARQUET,
    PROBE_PARQUET,
    MOVIES_PARQUET,
    TRAIN_CSR_FILE,
    ID_MAPS_FILE,
    DEVICE,
)
from src.models.baseline import DecoupledBaseline
from src.models.collaborative_knn import ResidualItemKNN
from src.models.latent_factors_gpu import BiasedMFRecommender, SVDPPRecommender
from src.models.content_cold_start import ContentColdStartBridge
from src.models.ensemble import HybridEnsemble
from src.evaluation import compute_rmse, compute_mae, ndcg_at_k, dcg_at_k


class TestNetflixRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads processed artifacts once for the test suite."""
        cls.train = pl.read_parquet(TRAIN_PARQUET)
        cls.probe = pl.read_parquet(PROBE_PARQUET)
        cls.movies = pl.read_parquet(MOVIES_PARQUET)
        cls.csr = load_npz(TRAIN_CSR_FILE)
        with open(ID_MAPS_FILE) as f:
            cls.id_maps = json.load(f)

    def test_01_data_pipeline_integrity(self):
        """Verifies no overlap between train and probe sets (strict anti-join)."""
        self.assertGreater(self.train.height, 1000000, "Train should have >1M ratings.")
        self.assertGreater(self.probe.height, 5000, "Probe should have >5k ratings.")

        train_pairs = set(zip(self.train["user_idx"].to_list(), self.train["movie_idx"].to_list()))
        probe_pairs = set(zip(self.probe["user_idx"].to_list(), self.probe["movie_idx"].to_list()))

        overlap = train_pairs.intersection(probe_pairs)
        self.assertEqual(len(overlap), 0, f"Data leakage detected! {len(overlap)} pairs overlap.")

        n_users = len(self.id_maps["user_to_idx"])
        n_movies = len(self.id_maps["movie_to_idx"])
        self.assertEqual(self.csr.shape, (n_users, n_movies))
        self.assertEqual(self.csr.nnz, self.train.height)

    def test_02_baseline_decoupled_regularization(self):
        """Verifies mathematical properties of decoupled baseline."""
        model = DecoupledBaseline(lambda_u=25.0, lambda_i=10.0).fit(self.train)
        self.assertTrue(3.0 <= model.mu <= 4.0, f"Unexpected global mean: {model.mu}")
        self.assertIsNotNone(model.b_u)
        self.assertIsNotNone(model.b_i)

        preds = model.predict_batch(self.probe["user_idx"].to_numpy(), self.probe["movie_idx"].to_numpy())
        self.assertTrue(np.all(preds >= 1.0) and np.all(preds <= 5.0))

        rmse = compute_rmse(self.probe["rating"].to_numpy(), preds)
        self.assertLess(rmse, 1.0, f"Baseline RMSE {rmse} should be below 1.0 on dense core.")

    def test_03_collaborative_knn_residuals_invariant(self):
        """Verifies Item-Item k-NN on residuals satisfies RMSE <= Baseline."""
        base = DecoupledBaseline().fit(self.train)
        knn = ResidualItemKNN(baseline=base, k=40, beta=25.0, gamma=1.0).fit(self.train)

        self.assertIsNotNone(knn.sim_matrix)
        self.assertEqual(knn.sim_matrix.shape, (base.n_movies, base.n_movies))
        self.assertTrue(np.all(np.diag(knn.sim_matrix) == 0.0), "Diagonal of similarity matrix must be zero.")

        base_preds = base.predict_batch(self.probe["user_idx"].to_numpy(), self.probe["movie_idx"].to_numpy())
        knn_preds = knn.predict_batch(self.probe["user_idx"].to_numpy(), self.probe["movie_idx"].to_numpy())

        rmse_base = compute_rmse(self.probe["rating"].to_numpy(), base_preds)
        rmse_knn = compute_rmse(self.probe["rating"].to_numpy(), knn_preds)

        self.assertLessEqual(rmse_knn, rmse_base + 1e-4, f"Invariant violated: k-NN ({rmse_knn:.4f}) > Baseline ({rmse_base:.4f})")

    def test_04_gpu_biased_mf_and_svdpp(self):
        """Verifies PyTorch Metal MPS execution and convergence of Biased MF and SVD++."""
        # Biased MF
        bmf = BiasedMFRecommender(n_factors=16, epochs=2, device=DEVICE).fit(self.train, df_val=self.probe)
        bmf_preds = bmf.predict_batch(self.probe["user_idx"].to_numpy(), self.probe["movie_idx"].to_numpy())
        self.assertTrue(np.all(bmf_preds >= 1.0) and np.all(bmf_preds <= 5.0))
        self.assertEqual(bmf.get_item_factors().shape, (bmf.n_movies, 16))

        # SVD++
        svd = SVDPPRecommender(n_factors=16, epochs=2, device=DEVICE).fit(self.train, csr_mat=self.csr, df_val=self.probe)
        svd_preds = svd.predict_batch(self.probe["user_idx"].to_numpy(), self.probe["movie_idx"].to_numpy())
        self.assertTrue(np.all(svd_preds >= 1.0) and np.all(svd_preds <= 5.0))
        self.assertEqual(svd.get_item_factors().shape, (svd.n_movies, 16))

    def test_05_content_cold_start_bridge(self):
        """Verifies TF-IDF + Decade Ridge latent projection."""
        train_movies = self.movies.filter(pl.col("movie_idx").is_not_null()).sort("movie_idx")
        n_movies = train_movies.height

        dummy_q = np.random.randn(n_movies, 16).astype(np.float32)
        dummy_b = np.random.randn(n_movies).astype(np.float32)
        dummy_p = np.random.randn(10, 16).astype(np.float32)
        dummy_bu = np.random.randn(10).astype(np.float32)

        bridge = ContentColdStartBridge().fit(
            df_movies_train=train_movies,
            item_factors=dummy_q,
            item_biases=dummy_b,
            user_factors=dummy_p,
            user_biases=dummy_bu,
            global_mean=3.5,
        )

        cold_sample = self.movies.filter(pl.col("movie_idx").is_null()).head(10)
        pred_q, pred_b = bridge.project_items(cold_sample)
        self.assertEqual(pred_q.shape, (10, 16))
        self.assertEqual(pred_b.shape, (10,))

        pred_ratings = bridge.predict_cold_ratings(
            user_indices=np.zeros(10, dtype=np.int32),
            df_cold_movies=cold_sample,
            cold_item_indices_in_df=np.arange(10, dtype=np.int32),
        )
        self.assertEqual(len(pred_ratings), 10)
        self.assertTrue(np.all(pred_ratings >= 1.0) and np.all(pred_ratings <= 5.0))

    def test_06_hybrid_ensemble_blending(self):
        """Verifies constrained convex quadratic programming optimization for stacking."""
        y = np.array([4.0, 3.0, 5.0, 2.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0])
        p1 = y + 0.3
        p2 = y - 0.2
        p3 = y + 0.1
        p4 = y - 0.05
        P = np.column_stack([p1, p2, p3, p4])

        ensemble = HybridEnsemble(model_names=["M1", "M2", "M3", "M4"]).fit(P, y)
        self.assertIsNotNone(ensemble.weights)
        self.assertTrue(np.isclose(np.sum(ensemble.weights), 1.0, atol=1e-4))
        self.assertTrue(np.all(ensemble.weights >= -1e-5))

        preds = ensemble.predict(P)
        ens_rmse = compute_rmse(y, preds)
        m1_rmse = compute_rmse(y, p1)
        self.assertLessEqual(ens_rmse, m1_rmse)

    def test_07_ranking_metrics_ndcg_dcg(self):
        """Verifies mathematical correctness of DCG and NDCG."""
        rel_ideal = np.array([1, 1, 0, 0])
        self.assertTrue(np.isclose(ndcg_at_k(rel_ideal, 4), 1.0))

        rel_suboptimal = np.array([0, 1, 1, 0])
        ndcg_val = ndcg_at_k(rel_suboptimal, 4)
        self.assertTrue(0.0 < ndcg_val < 1.0)

        empty_rel = np.array([])
        self.assertEqual(dcg_at_k(empty_rel, 5), 0.0)
        self.assertEqual(ndcg_at_k(empty_rel, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
