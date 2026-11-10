"""
Item-Item Collaborative Filtering on Residuals with Pearson Correlation and Bayesian Shrinkage.
Accelerated using SciPy CSR sparse matrix operations and Numba JIT parallelism.

Formulation (Bell & Koren 2007, Koren 2008):
  e_ui = r_ui - b_ui
  rho_ij = sum_{u in U(i,j)}(e_ui * e_uj) / (sqrt(sum e_ui^2) * sqrt(sum e_uj^2) + eps)
  s_ij = max(0, (n_ij / (n_ij + beta)) * rho_ij)
  r_hat_ui = b_ui + (sum_{j in S^k(i; u)} s_ij * e_uj) / (sum_{j in S^k(i; u)} |s_ij| + gamma)
"""

import time
import logging
from typing import Optional, Tuple
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, csc_matrix
from numba import njit, prange

from src.config import KNN_CONFIG
from src.models.baseline import DecoupledBaseline

logger = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True)
def _predict_knn_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    sim: np.ndarray,
    probe_u: np.ndarray,
    probe_m: np.ndarray,
    base_preds: np.ndarray,
    k: int,
    gamma: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """
    Numba JIT accelerated prediction kernel with multi-core parallelism.
    """
    n_samples = len(probe_u)
    out = np.empty(n_samples, dtype=np.float32)

    for idx in prange(n_samples):
        u = probe_u[idx]
        target_m = probe_m[idx]
        b_ui = base_preds[idx]

        if u < 0 or u >= len(indptr) - 1:
            out[idx] = b_ui
            continue

        start = indptr[u]
        end = indptr[u + 1]
        n_rated = end - start

        if n_rated == 0 or target_m < 0 or target_m >= sim.shape[0]:
            out[idx] = b_ui
            continue

        s_vals = np.empty(n_rated, dtype=np.float32)
        e_vals = np.empty(n_rated, dtype=np.float32)
        pos_count = 0

        for i in range(start, end):
            rated_item = indices[i]
            s = sim[target_m, rated_item]
            if s > 0.0:
                s_vals[pos_count] = s
                e_vals[pos_count] = data[i]
                pos_count += 1

        if pos_count == 0:
            out[idx] = b_ui
            continue

        num = 0.0
        den = 0.0

        if pos_count <= k:
            for j in range(pos_count):
                num += s_vals[j] * e_vals[j]
                den += s_vals[j]
        else:
            # Selection of top k similarities via partial selection
            for j in range(k):
                max_pos = j
                for m in range(j + 1, pos_count):
                    if s_vals[m] > s_vals[max_pos]:
                        max_pos = m

                tmp_s = s_vals[j]
                s_vals[j] = s_vals[max_pos]
                s_vals[max_pos] = tmp_s

                tmp_e = e_vals[j]
                e_vals[j] = e_vals[max_pos]
                e_vals[max_pos] = tmp_e

                num += s_vals[j] * e_vals[j]
                den += s_vals[j]

        pred = b_ui + (num / (den + gamma))
        if pred < clip_min:
            pred = clip_min
        elif pred > clip_max:
            pred = clip_max

        out[idx] = pred

    return out


class ResidualItemKNN:
    """
    Item-Item Collaborative Filtering on Residuals with Bayesian Shrinkage.
    """

    def __init__(
        self,
        baseline: Optional[DecoupledBaseline] = None,
        k: int = KNN_CONFIG["k"],
        beta: float = KNN_CONFIG["beta"],
        gamma: float = KNN_CONFIG["gamma"],
        clip_min: float = KNN_CONFIG["clip_min"],
        clip_max: float = KNN_CONFIG["clip_max"],
    ):
        self.baseline = baseline or DecoupledBaseline()
        self.k = int(k)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self.sim_matrix: Optional[np.ndarray] = None
        self.residual_csr: Optional[csr_matrix] = None
        self.n_users: int = 0
        self.n_movies: int = 0

    def fit(self, df_train: pl.DataFrame, n_users: Optional[int] = None, n_movies: Optional[int] = None) -> "ResidualItemKNN":
        """
        Fits baseline, calculates residuals, and computes the Bayesian shrunk item similarity matrix.
        """
        logger.info("Fitting baseline model for residual calculation...")
        self.baseline.fit(df_train, n_users=n_users, n_movies=n_movies)
        self.n_users = self.baseline.n_users
        self.n_movies = self.baseline.n_movies

        residuals = self.baseline.compute_residuals(df_train)
        u_arr = df_train["user_idx"].to_numpy().astype(np.int32)
        m_arr = df_train["movie_idx"].to_numpy().astype(np.int32)

        logger.info(f"Building residual CSR matrix ({self.n_users} x {self.n_movies})...")
        self.residual_csr = csr_matrix((residuals, (u_arr, m_arr)), shape=(self.n_users, self.n_movies), dtype=np.float32)
        E_csc = self.residual_csr.tocsc()

        # Binary CSR for co-occurrence counting
        B_csc = csr_matrix((np.ones(len(df_train), dtype=np.float32), (u_arr, m_arr)), shape=(self.n_users, self.n_movies), dtype=np.float32).tocsc()

        t0 = time.time()
        logger.info("Computing Item-Item similarity matrix with Bayesian shrinkage...")

        # Dot products between item residual columns
        dot_product = (E_csc.T @ E_csc).toarray().astype(np.float32)

        # Co-occurrence counts
        co_counts = (B_csc.T @ B_csc).toarray().astype(np.float32)

        # L2 column norms
        col_norms = np.sqrt(np.array(E_csc.multiply(E_csc).sum(axis=0)).flatten()).astype(np.float32)
        denom = np.outer(col_norms, col_norms) + 1e-9

        # Pearson correlation on residuals
        corr = dot_product / denom

        # Bayesian shrinkage: s_ij = (n_ij / (n_ij + beta)) * rho_ij
        shrinkage = co_counts / (co_counts + self.beta)
        sim = np.maximum(0.0, corr * shrinkage)
        np.fill_diagonal(sim, 0.0)

        self.sim_matrix = sim
        logger.info(f"Item similarity matrix ({self.n_movies}x{self.n_movies}) computed in {time.time() - t0:.2f}s.")
        return self

    def predict_batch(self, user_indices: np.ndarray, movie_indices: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated batch prediction for user and movie arrays.
        """
        assert self.sim_matrix is not None and self.residual_csr is not None, "Model must be fitted."

        u_idx = np.asarray(user_indices, dtype=np.int32)
        m_idx = np.asarray(movie_indices, dtype=np.int32)

        # Baseline predictions
        base_preds = self.baseline.predict_batch(u_idx, m_idx)

        # Parallel Numba execution
        preds = _predict_knn_numba(
            self.residual_csr.indptr.astype(np.int32),
            self.residual_csr.indices.astype(np.int32),
            self.residual_csr.data.astype(np.float32),
            self.sim_matrix,
            u_idx,
            m_idx,
            base_preds,
            self.k,
            self.gamma,
            self.clip_min,
            self.clip_max,
        )
        return preds

    def get_similar_items(self, movie_idx: int, top_n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Returns top_n most similar movie indices and their similarity scores."""
        if self.sim_matrix is None or movie_idx < 0 or movie_idx >= self.n_movies:
            return np.array([]), np.array([])
        sims = self.sim_matrix[movie_idx]
        top_indices = np.argsort(sims)[::-1][:top_n]
        return top_indices, sims[top_indices]
