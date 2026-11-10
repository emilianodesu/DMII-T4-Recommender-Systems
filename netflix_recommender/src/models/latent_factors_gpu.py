"""
Latent Factor Models for Collaborative Filtering accelerated on Apple Silicon GPU (Metal Performance Shaders - MPS).
Implements:
1. Biased Matrix Factorization (Biased MF):
     r_hat_ui = mu + b_u + b_i + p_u^T q_i
2. SVD++ (Koren 2008) with Implicit Feedback:
     r_hat_ui = mu + b_u + b_i + q_i^T (p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j)
"""

import time
import logging
from typing import Optional, Dict, List, Tuple
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, load_npz
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import DEVICE, BIASED_MF_CONFIG, SVDPP_CONFIG, TRAIN_CSR_FILE

logger = logging.getLogger(__name__)


# ============================================================================
# PyTorch Modules
# ============================================================================

class BiasedMFNet(nn.Module):
    """
    PyTorch Neural Network Module for Biased Matrix Factorization.
    """

    def __init__(self, n_users: int, n_movies: int, n_factors: int = 32, mu: float = 3.6):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32), requires_grad=False)
        self.b_u = nn.Embedding(n_users, 1)
        self.b_i = nn.Embedding(n_movies, 1)
        self.p_u = nn.Embedding(n_users, n_factors)
        self.q_i = nn.Embedding(n_movies, n_factors)

        # Initialize weights
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)
        std = 0.05 / np.sqrt(n_factors)
        nn.init.normal_(self.p_u.weight, mean=0.0, std=std)
        nn.init.normal_(self.q_i.weight, mean=0.0, std=std)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        bu = self.b_u(u).squeeze(-1)
        bi = self.b_i(i).squeeze(-1)
        dot = (self.p_u(u) * self.q_i(i)).sum(dim=-1)
        return self.mu + bu + bi + dot


class SVDPPNet(nn.Module):
    """
    PyTorch Neural Network Module for SVD++ (Koren 2008).
    Incorporates implicit feedback from user history via EmbeddingBag.
    """

    def __init__(self, n_users: int, n_movies: int, n_factors: int = 32, mu: float = 3.6):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32), requires_grad=False)
        self.b_u = nn.Embedding(n_users, 1)
        self.b_i = nn.Embedding(n_movies, 1)
        self.p_u = nn.Embedding(n_users, n_factors)
        self.q_i = nn.Embedding(n_movies, n_factors)
        self.y_j = nn.EmbeddingBag(n_movies, n_factors, mode="sum")

        # Initialize weights
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)
        std = 0.05 / np.sqrt(n_factors)
        nn.init.normal_(self.p_u.weight, mean=0.0, std=std)
        nn.init.normal_(self.q_i.weight, mean=0.0, std=std)
        nn.init.normal_(self.y_j.weight, mean=0.0, std=std)

    def forward(
        self,
        u: torch.Tensor,
        i: torch.Tensor,
        hist_indices: torch.Tensor,
        hist_offsets: torch.Tensor,
        hist_norm: torch.Tensor,
    ) -> torch.Tensor:
        # User implicit preference representation
        c = self.y_j(hist_indices, hist_offsets) * hist_norm
        u_lat = self.p_u(u) + c[u]
        i_lat = self.q_i(i)

        bu = self.b_u(u).squeeze(-1)
        bi = self.b_i(i).squeeze(-1)
        dot = (u_lat * i_lat).sum(dim=-1)
        return self.mu + bu + bi + dot


# ============================================================================
# Recommender Estimators
# ============================================================================

class BiasedMFRecommender:
    """
    Estimator wrapper for Biased Matrix Factorization running on Apple Silicon GPU MPS.
    """

    def __init__(
        self,
        n_factors: int = BIASED_MF_CONFIG["n_factors"],
        lr: float = BIASED_MF_CONFIG["lr"],
        weight_decay: float = BIASED_MF_CONFIG["weight_decay"],
        epochs: int = BIASED_MF_CONFIG["epochs"],
        batch_size: int = BIASED_MF_CONFIG["batch_size"],
        clip_min: float = BIASED_MF_CONFIG["clip_min"],
        clip_max: float = BIASED_MF_CONFIG["clip_max"],
        device: torch.device = DEVICE,
    ):
        self.n_factors = n_factors
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device

        self.model: Optional[BiasedMFNet] = None
        self.n_users: int = 0
        self.n_movies: int = 0
        self.mu: float = 0.0
        self.history: List[Dict[str, float]] = []

    def fit(
        self,
        df_train: pl.DataFrame,
        df_val: Optional[pl.DataFrame] = None,
        n_users: Optional[int] = None,
        n_movies: Optional[int] = None,
    ) -> "BiasedMFRecommender":
        self.n_users = n_users if n_users is not None else int(df_train["user_idx"].max()) + 1
        self.n_movies = n_movies if n_movies is not None else int(df_train["movie_idx"].max()) + 1
        self.mu = float(df_train["rating"].mean())

        logger.info(f"Initializing Biased MF on {self.device} ({self.n_users} users, {self.n_movies} movies, {self.n_factors} factors)...")
        self.model = BiasedMFNet(self.n_users, self.n_movies, self.n_factors, self.mu).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        train_u = torch.tensor(df_train["user_idx"].to_numpy(), dtype=torch.long, device=self.device)
        train_i = torch.tensor(df_train["movie_idx"].to_numpy(), dtype=torch.long, device=self.device)
        train_r = torch.tensor(df_train["rating"].to_numpy(), dtype=torch.float32, device=self.device)
        n_train = len(train_u)

        if df_val is not None:
            val_u = torch.tensor(df_val["user_idx"].to_numpy(), dtype=torch.long, device=self.device)
            val_i = torch.tensor(df_val["movie_idx"].to_numpy(), dtype=torch.long, device=self.device)
            val_r = df_val["rating"].to_numpy()

        self.history = []
        best_state = None
        best_val = float("inf")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.model.train()
            perm = torch.randperm(n_train, device=self.device)
            total_loss = 0.0

            for b in range(0, n_train, self.batch_size):
                batch_idx = perm[b : b + self.batch_size]
                optimizer.zero_grad()
                preds = self.model(train_u[batch_idx], train_i[batch_idx])
                loss = criterion(preds, train_r[batch_idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)

            train_loss = total_loss / n_train
            val_rmse = None

            if df_val is not None:
                self.model.eval()
                with torch.no_grad():
                    v_preds = self.model(val_u, val_i).clamp(self.clip_min, self.clip_max).cpu().numpy()
                    val_rmse = float(np.sqrt(np.mean((val_r - v_preds) ** 2)))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            elapsed = time.time() - t0
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse, "time_s": elapsed})
            logger.info(f"Biased MF Epoch {epoch:2d}/{self.epochs} - Loss: {train_loss:.4f} - Val RMSE: {val_rmse:.4f} ({elapsed:.2f}s)")

        if best_state is not None:
            logger.info(f"Restoring best Biased MF weights with Val RMSE: {best_val:.4f}")
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def predict_batch(self, user_indices: np.ndarray, movie_indices: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model must be fitted."
        self.model.eval()
        u_arr = np.asarray(user_indices, dtype=np.int64)
        m_arr = np.asarray(movie_indices, dtype=np.int64)

        # Handle out of range by mapping to 0 and later masking
        valid_mask = (u_arr >= 0) & (u_arr < self.n_users) & (m_arr >= 0) & (m_arr < self.n_movies)
        safe_u = np.where(valid_mask, u_arr, 0)
        safe_m = np.where(valid_mask, m_arr, 0)

        u_tensor = torch.tensor(safe_u, dtype=torch.long, device=self.device)
        m_tensor = torch.tensor(safe_m, dtype=torch.long, device=self.device)

        with torch.no_grad():
            preds = self.model(u_tensor, m_tensor).clamp(self.clip_min, self.clip_max).cpu().numpy()

        preds[~valid_mask] = self.mu
        return preds

    def get_item_factors(self) -> np.ndarray:
        assert self.model is not None, "Model must be fitted."
        return self.model.q_i.weight.detach().cpu().numpy()

    def get_user_factors(self) -> np.ndarray:
        assert self.model is not None, "Model must be fitted."
        return self.model.p_u.weight.detach().cpu().numpy()

    def get_item_biases(self) -> np.ndarray:
        assert self.model is not None, "Model must be fitted."
        return self.model.b_i.weight.detach().squeeze(-1).cpu().numpy()


class SVDPPRecommender:
    """
    Estimator wrapper for SVD++ (Koren 2008) running on Apple Silicon GPU MPS.
    """

    def __init__(
        self,
        n_factors: int = SVDPP_CONFIG["n_factors"],
        lr: float = SVDPP_CONFIG["lr"],
        weight_decay: float = SVDPP_CONFIG["weight_decay"],
        epochs: int = SVDPP_CONFIG["epochs"],
        batch_size: int = SVDPP_CONFIG["batch_size"],
        clip_min: float = SVDPP_CONFIG["clip_min"],
        clip_max: float = SVDPP_CONFIG["clip_max"],
        device: torch.device = DEVICE,
    ):
        self.n_factors = n_factors
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device

        self.model: Optional[SVDPPNet] = None
        self.n_users: int = 0
        self.n_movies: int = 0
        self.mu: float = 0.0
        self.history: List[Dict[str, float]] = []

        self.hist_indices: Optional[torch.Tensor] = None
        self.hist_offsets: Optional[torch.Tensor] = None
        self.hist_norm: Optional[torch.Tensor] = None

    def _prepare_csr_tensors(self, csr_mat: csr_matrix):
        """Prepares user rating history tensors for MPS EmbeddingBag."""
        indptr = torch.tensor(csr_mat.indptr[:-1], dtype=torch.long, device=self.device)
        indices = torch.tensor(csr_mat.indices, dtype=torch.long, device=self.device)
        counts = torch.diff(torch.tensor(csr_mat.indptr, dtype=torch.float32, device=self.device))
        norm = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts)).unsqueeze(1)

        self.hist_offsets = indptr
        self.hist_indices = indices
        self.hist_norm = norm

    def fit(
        self,
        df_train: pl.DataFrame,
        csr_mat: Optional[csr_matrix] = None,
        df_val: Optional[pl.DataFrame] = None,
        n_users: Optional[int] = None,
        n_movies: Optional[int] = None,
    ) -> "SVDPPRecommender":
        self.n_users = n_users if n_users is not None else int(df_train["user_idx"].max()) + 1
        self.n_movies = n_movies if n_movies is not None else int(df_train["movie_idx"].max()) + 1
        self.mu = float(df_train["rating"].mean())

        if csr_mat is None:
            if TRAIN_CSR_FILE.exists():
                csr_mat = load_npz(TRAIN_CSR_FILE)
            else:
                u_arr = df_train["user_idx"].to_numpy()
                m_arr = df_train["movie_idx"].to_numpy()
                r_arr = df_train["rating"].to_numpy()
                csr_mat = csr_matrix((r_arr, (u_arr, m_arr)), shape=(self.n_users, self.n_movies), dtype=np.float32)

        self._prepare_csr_tensors(csr_mat)

        logger.info(f"Initializing SVD++ on {self.device} ({self.n_users} users, {self.n_movies} movies, {self.n_factors} factors)...")
        self.model = SVDPPNet(self.n_users, self.n_movies, self.n_factors, self.mu).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        train_u = torch.tensor(df_train["user_idx"].to_numpy(), dtype=torch.long, device=self.device)
        train_i = torch.tensor(df_train["movie_idx"].to_numpy(), dtype=torch.long, device=self.device)
        train_r = torch.tensor(df_train["rating"].to_numpy(), dtype=torch.float32, device=self.device)
        n_train = len(train_u)

        if df_val is not None:
            val_u = torch.tensor(df_val["user_idx"].to_numpy(), dtype=torch.long, device=self.device)
            val_i = torch.tensor(df_val["movie_idx"].to_numpy(), dtype=torch.long, device=self.device)
            val_r = df_val["rating"].to_numpy()

        self.history = []
        best_state = None
        best_val = float("inf")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.model.train()
            perm = torch.randperm(n_train, device=self.device)
            total_loss = 0.0

            for b in range(0, n_train, self.batch_size):
                batch_idx = perm[b : b + self.batch_size]
                optimizer.zero_grad()
                preds = self.model(
                    train_u[batch_idx],
                    train_i[batch_idx],
                    self.hist_indices,
                    self.hist_offsets,
                    self.hist_norm,
                )
                loss = criterion(preds, train_r[batch_idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)

            train_loss = total_loss / n_train
            val_rmse = None

            if df_val is not None:
                self.model.eval()
                with torch.no_grad():
                    v_preds = self.model(
                        val_u,
                        val_i,
                        self.hist_indices,
                        self.hist_offsets,
                        self.hist_norm,
                    ).clamp(self.clip_min, self.clip_max).cpu().numpy()
                    val_rmse = float(np.sqrt(np.mean((val_r - v_preds) ** 2)))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            elapsed = time.time() - t0
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse, "time_s": elapsed})
            logger.info(f"SVD++ Epoch {epoch:2d}/{self.epochs} - Loss: {train_loss:.4f} - Val RMSE: {val_rmse:.4f} ({elapsed:.2f}s)")

        if best_state is not None:
            logger.info(f"Restoring best SVD++ weights with Val RMSE: {best_val:.4f}")
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def predict_batch(self, user_indices: np.ndarray, movie_indices: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model must be fitted."
        self.model.eval()
        u_arr = np.asarray(user_indices, dtype=np.int64)
        m_arr = np.asarray(movie_indices, dtype=np.int64)

        valid_mask = (u_arr >= 0) & (u_arr < self.n_users) & (m_arr >= 0) & (m_arr < self.n_movies)
        safe_u = np.where(valid_mask, u_arr, 0)
        safe_m = np.where(valid_mask, m_arr, 0)

        u_tensor = torch.tensor(safe_u, dtype=torch.long, device=self.device)
        m_tensor = torch.tensor(safe_m, dtype=torch.long, device=self.device)

        with torch.no_grad():
            preds = self.model(
                u_tensor,
                m_tensor,
                self.hist_indices,
                self.hist_offsets,
                self.hist_norm,
            ).clamp(self.clip_min, self.clip_max).cpu().numpy()

        preds[~valid_mask] = self.mu
        return preds

    def get_item_factors(self) -> np.ndarray:
        assert self.model is not None, "Model must be fitted."
        return self.model.q_i.weight.detach().cpu().numpy()

    def get_user_factors(self) -> np.ndarray:
        assert self.model is not None, "Model must be fitted."
        return self.model.p_u.weight.detach().cpu().numpy()

    def get_item_biases(self) -> np.ndarray:
        assert self.model is not None, "Model must be fitted."
        return self.model.b_i.weight.detach().squeeze(-1).cpu().numpy()
