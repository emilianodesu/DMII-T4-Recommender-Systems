"""
Content-Based Cold-Start Recommendation via TF-IDF and Ridge Latent Projection Bridge.
Solves the Cold-Start problem for movies without historical ratings by mapping
metadata (movie titles + release decades) into the learned latent factor space (SVD++ / Biased MF).

Formulation:
  X_{content} = [TF-IDF(Title), OneHot(Decade)]
  W_q^* = argmin_W ||X W - Q_{train}||_F^2 + alpha ||W||_F^2
  w_b^* = argmin_w ||X w - b_{train}||_2^2 + alpha ||w||_2^2
  q_hat_{cold} = X_{cold} W_q^*
  b_hat_{cold} = X_{cold} w_b^*
  r_hat_{u, cold} = mu + b_u + b_hat_{cold} + p_u^T q_hat_{cold}
"""

import logging
from typing import Optional, Tuple, Dict, List
import numpy as np
import polars as pl
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

from src.config import COLD_START_CONFIG, MOVIES_PARQUET

logger = logging.getLogger(__name__)


class ContentColdStartBridge:
    """
    Projects movie content features (title TF-IDF + release decade)
    into learned latent item factor embeddings and biases using Ridge Regression.
    """

    def __init__(
        self,
        ridge_alpha: float = COLD_START_CONFIG["ridge_alpha"],
        tfidf_max_features: int = COLD_START_CONFIG["tfidf_max_features"],
        ngram_range: Tuple[int, int] = COLD_START_CONFIG["ngram_range"],
    ):
        self.ridge_alpha = float(ridge_alpha)
        self.tfidf_max_features = int(tfidf_max_features)
        self.ngram_range = ngram_range

        self.tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        self.ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")

        self.ridge_q = Ridge(alpha=self.ridge_alpha)
        self.ridge_b = Ridge(alpha=self.ridge_alpha)

        self.is_fitted: bool = False
        self.mu: float = 3.6
        self.user_factors: Optional[np.ndarray] = None
        self.user_biases: Optional[np.ndarray] = None

    def _extract_content_features(self, df_movies: pl.DataFrame, fit_transformers: bool = False) -> csr_matrix:
        """Extracts combined TF-IDF and decade sparse features."""
        titles = df_movies["title"].fill_null("").to_list()

        if fit_transformers:
            X_text = self.tfidf.fit_transform(titles)
        else:
            X_text = self.tfidf.transform(titles)

        # Release decade
        years = df_movies["year"].fill_null(2000).to_numpy()
        decades = (years // 10 * 10).reshape(-1, 1)

        if fit_transformers:
            X_dec = self.ohe.fit_transform(decades)
        else:
            X_dec = self.ohe.transform(decades)

        return hstack([X_text, X_dec]).tocsr()

    def fit(
        self,
        df_movies_train: pl.DataFrame,
        item_factors: np.ndarray,
        item_biases: np.ndarray,
        user_factors: np.ndarray,
        user_biases: np.ndarray,
        global_mean: float,
    ) -> "ContentColdStartBridge":
        """
        Fits content feature extractors and the Ridge regression mapping to latent factors.
        df_movies_train: DataFrame containing movies present in training set, sorted by movie_idx.
        """
        logger.info(f"Fitting Content Cold-Start Bridge for {df_movies_train.height} items...")
        self.mu = float(global_mean)
        self.user_factors = user_factors
        self.user_biases = user_biases

        X_train = self._extract_content_features(df_movies_train, fit_transformers=True)

        logger.info(f"Extracted {X_train.shape[1]} content features (TF-IDF + Decades). Training Ridge Bridge...")
        self.ridge_q.fit(X_train, item_factors)
        self.ridge_b.fit(X_train, item_biases)
        self.is_fitted = True

        r2_q = self.ridge_q.score(X_train, item_factors)
        r2_b = self.ridge_b.score(X_train, item_biases)
        logger.info(f"Ridge Bridge fitted: R^2 on Factors = {r2_q:.4f}, R^2 on Biases = {r2_b:.4f}.")
        return self

    def project_items(self, df_cold_movies: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts latent factor embeddings and item biases for unseen/cold-start movies.
        """
        assert self.is_fitted, "Bridge must be fitted before project_items."
        X_cold = self._extract_content_features(df_cold_movies, fit_transformers=False)
        pred_q = self.ridge_q.predict(X_cold)
        pred_b = self.ridge_b.predict(X_cold)
        return pred_q, pred_b

    def predict_cold_ratings(
        self,
        user_indices: np.ndarray,
        df_cold_movies: pl.DataFrame,
        cold_item_indices_in_df: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts ratings for given users on cold-start movies using projected latent factors.
        """
        assert self.is_fitted and self.user_factors is not None, "Bridge must be fitted."
        pred_q, pred_b = self.project_items(df_cold_movies)

        u_idx = np.asarray(user_indices, dtype=np.int32)
        cold_idx = np.asarray(cold_item_indices_in_df, dtype=np.int32)

        # User biases & factors
        n_users = len(self.user_factors)
        valid_u = (u_idx >= 0) & (u_idx < n_users)
        safe_u = np.where(valid_u, u_idx, 0)

        bu = np.where(valid_u, self.user_biases[safe_u], 0.0)
        pu = self.user_factors[safe_u]

        bi = pred_b[cold_idx]
        qi = pred_q[cold_idx]

        dot = np.sum(pu * qi, axis=1)
        preds = self.mu + bu + bi + dot
        return np.clip(preds, 1.0, 5.0)

    def evaluate_cold_start_simulation(
        self,
        df_all_ratings: pl.DataFrame,
        df_movies: pl.DataFrame,
        n_cold_movies: int = COLD_START_CONFIG["simulated_cold_movies"],
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Simulates cold-start scenario:
        1. Selects n_cold_movies randomly.
        2. Predicts their ratings using Content Bridge vs Baseline/Global Mean.
        3. Computes RMSE improvement.
        """
        np.random.seed(seed)
        sampled_mids = df_movies.filter(pl.col("movie_idx").is_not_null())["movie_id"].to_list()
        cold_mids = set(np.random.choice(sampled_mids, size=min(n_cold_movies, len(sampled_mids)), replace=False))

        # Filter test ratings on cold movies
        cold_ratings = df_all_ratings.filter(pl.col("movie_id").is_in(cold_mids))
        if cold_ratings.height == 0:
            return {"rmse_bridge": 0.0, "rmse_global_mean": 0.0}

        cold_movies_df = df_movies.filter(pl.col("movie_id").is_in(cold_mids)).sort("movie_id")
        mid_to_cold_row = {mid: idx for idx, mid in enumerate(cold_movies_df["movie_id"].to_list())}

        test_u = cold_ratings["user_idx"].to_numpy()
        test_m_ids = cold_ratings["movie_id"].to_numpy()
        test_cold_rows = np.array([mid_to_cold_row[mid] for mid in test_m_ids])
        actual_r = cold_ratings["rating"].to_numpy()

        pred_bridge = self.predict_cold_ratings(test_u, cold_movies_df, test_cold_rows)
        pred_mean = np.full_like(actual_r, fill_value=self.mu)

        rmse_bridge = float(np.sqrt(np.mean((actual_r - pred_bridge) ** 2)))
        rmse_mean = float(np.sqrt(np.mean((actual_r - pred_mean) ** 2)))

        return {
            "n_cold_movies": len(cold_mids),
            "n_test_ratings": cold_ratings.height,
            "rmse_bridge": rmse_bridge,
            "rmse_global_mean": rmse_mean,
            "rmse_improvement": rmse_mean - rmse_bridge,
        }
