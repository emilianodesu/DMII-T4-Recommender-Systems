"""
Decoupled Regularized Baseline Model for Netflix Prize Recommender.
Implements the Bell & Koren (2007) formulation:
  r_hat_ui = mu + b_u + b_i
where:
  mu  = global mean rating
  b_i = sum_{u in R(i)}(r_ui - mu) / (|R(i)| + lambda_i)
  b_u = sum_{i in R(u)}(r_ui - mu - b_i) / (|R(u)| + lambda_u)
"""

import numpy as np
import polars as pl
from typing import Tuple, Optional
from src.config import BASELINE_CONFIG


class DecoupledBaseline:
    """
    Decoupled Regularized Baseline Recommender.
    Solves for item and user biases using separate L2 regularization shrinkage parameters.
    """

    def __init__(
        self,
        lambda_u: float = BASELINE_CONFIG["lambda_u"],
        lambda_i: float = BASELINE_CONFIG["lambda_i"],
        clip_min: float = BASELINE_CONFIG["clip_min"],
        clip_max: float = BASELINE_CONFIG["clip_max"],
    ):
        self.lambda_u = float(lambda_u)
        self.lambda_i = float(lambda_i)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self.mu: float = 0.0
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None
        self.n_users: int = 0
        self.n_movies: int = 0

    def fit(self, df_train: pl.DataFrame, n_users: Optional[int] = None, n_movies: Optional[int] = None) -> "DecoupledBaseline":
        """
        Fits global mean, item biases b_i, and decoupled user biases b_u.
        """
        if n_users is None:
            self.n_users = int(df_train["user_idx"].max()) + 1
        else:
            self.n_users = n_users

        if n_movies is None:
            self.n_movies = int(df_train["movie_idx"].max()) + 1
        else:
            self.n_movies = n_movies

        # 1. Global Mean mu
        self.mu = float(df_train["rating"].mean())

        # 2. Movie biases b_i
        # b_i = sum(r_ui - mu) / (|R(i)| + lambda_i)
        movie_stats = df_train.group_by("movie_idx").agg([
            (pl.col("rating") - self.mu).sum().alias("diff_sum"),
            pl.len().alias("count"),
        ])

        self.b_i = np.zeros(self.n_movies, dtype=np.float32)
        for row in movie_stats.iter_rows(named=True):
            m_idx = row["movie_idx"]
            if m_idx < self.n_movies:
                self.b_i[m_idx] = row["diff_sum"] / (row["count"] + self.lambda_i)

        # 3. User biases b_u
        # b_u = sum(r_ui - mu - b_i) / (|R(u)| + lambda_u)
        train_movie_indices = df_train["movie_idx"].to_numpy()
        b_i_vals = self.b_i[train_movie_indices]

        train_augmented = df_train.with_columns(
            pl.Series("b_i", b_i_vals)
        )

        user_stats = train_augmented.group_by("user_idx").agg([
            (pl.col("rating") - self.mu - pl.col("b_i")).sum().alias("diff_sum"),
            pl.len().alias("count"),
        ])

        self.b_u = np.zeros(self.n_users, dtype=np.float32)
        for row in user_stats.iter_rows(named=True):
            u_idx = row["user_idx"]
            if u_idx < self.n_users:
                self.b_u[u_idx] = row["diff_sum"] / (row["count"] + self.lambda_u)

        return self

    def predict_batch(self, user_indices: np.ndarray, movie_indices: np.ndarray) -> np.ndarray:
        """
        Vectorized prediction for arrays of user and movie indices.
        Handles out-of-vocabulary indices by defaulting unknown biases to 0.0.
        """
        assert self.b_u is not None and self.b_i is not None, "Model must be fitted before predict."

        u_idx = np.asarray(user_indices, dtype=np.int32)
        m_idx = np.asarray(movie_indices, dtype=np.int32)

        # Mask in-bounds indices
        valid_u = (u_idx >= 0) & (u_idx < self.n_users)
        valid_m = (m_idx >= 0) & (m_idx < self.n_movies)

        bu_vals = np.zeros(len(u_idx), dtype=np.float32)
        bi_vals = np.zeros(len(m_idx), dtype=np.float32)

        bu_vals[valid_u] = self.b_u[u_idx[valid_u]]
        bi_vals[valid_m] = self.b_i[m_idx[valid_m]]

        preds = self.mu + bu_vals + bi_vals
        return np.clip(preds, self.clip_min, self.clip_max)

    def predict(self, user_idx: int, movie_idx: int) -> float:
        """Single user-item rating prediction."""
        u_b = self.b_u[user_idx] if 0 <= user_idx < self.n_users else 0.0
        m_b = self.b_i[movie_idx] if 0 <= movie_idx < self.n_movies else 0.0
        pred = self.mu + u_b + m_b
        return float(np.clip(pred, self.clip_min, self.clip_max))

    def compute_residuals(self, df: pl.DataFrame) -> np.ndarray:
        """
        Computes residual e_ui = r_ui - b_ui for collaborative filtering training.
        """
        preds = self.predict_batch(df["user_idx"].to_numpy(), df["movie_idx"].to_numpy())
        ratings = df["rating"].to_numpy()
        return (ratings - preds).astype(np.float32)
