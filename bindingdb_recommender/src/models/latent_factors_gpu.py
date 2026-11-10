"""Latent Factor Recommendation Models on GPU (PyTorch Metal MPS / CUDA).
Implements:
1) Biased Matrix Factorization:
   y_hat(t, l) = mu + b_t + b_l + <p_t, q_l>
2) Two-Tower Hybrid Neural Matrix Factorization:
   q_l = (1 - delta_l) * e_l + MLP_chem(ECFP4_l)
   p_t = (1 - delta_t) * d_t + MLP_prot(Seq_t)
   with Embedding Dropout (delta ~ Bernoulli(p)) to bridge Warm and Cold-Start regimes.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import LatentFactorsGPUConfig, TwoTowerHybridConfig
from src.featurizer import MoleculeFeaturizer, ProteinFeaturizer


# ---------------------------------------------------------------------------
# 1. Biased Matrix Factorization
# ---------------------------------------------------------------------------

class BiasedMFNet(nn.Module):
    """PyTorch Biased Matrix Factorization Architecture."""

    def __init__(self, n_targets: int, n_ligands: int, latent_dim: int = 64, global_mean: float = 6.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.mu = nn.Parameter(torch.tensor(float(global_mean), dtype=torch.float32), requires_grad=True)

        # Index 0 is reserved for unknown / cold entities (padding_idx=0)
        self.target_bias = nn.Embedding(n_targets + 1, 1, padding_idx=0)
        self.ligand_bias = nn.Embedding(n_ligands + 1, 1, padding_idx=0)

        self.target_factors = nn.Embedding(n_targets + 1, latent_dim, padding_idx=0)
        self.ligand_factors = nn.Embedding(n_ligands + 1, latent_dim, padding_idx=0)

        # Initialization
        nn.init.zeros_(self.target_bias.weight)
        nn.init.zeros_(self.ligand_bias.weight)
        nn.init.normal_(self.target_factors.weight, std=0.05)
        nn.init.normal_(self.ligand_factors.weight, std=0.05)
        with torch.no_grad():
            self.target_factors.weight[0].fill_(0.0)
            self.ligand_factors.weight[0].fill_(0.0)

    def forward(self, t_indices: torch.Tensor, l_indices: torch.Tensor) -> torch.Tensor:
        b_t = self.target_bias(t_indices).squeeze(-1)
        b_l = self.ligand_bias(l_indices).squeeze(-1)
        p_t = self.target_factors(t_indices)
        q_l = self.ligand_factors(l_indices)

        interaction = (p_t * q_l).sum(dim=-1)
        return self.mu + b_t + b_l + interaction


class BiasedMatrixFactorization:
    """Biased Matrix Factorization Recommender with GPU hardware acceleration."""

    def __init__(self, config: Optional[LatentFactorsGPUConfig] = None):
        self.config = config or LatentFactorsGPUConfig()
        self.device = torch.device(self.config.device)
        self.model: Optional[BiasedMFNet] = None
        self.target_to_idx: Dict[str, int] = {}
        self.ligand_to_idx: Dict[str, int] = {}
        self.global_mean: float = 6.0
        self.is_fitted: bool = False

    def fit(self, train_df: pl.DataFrame, val_df: Optional[pl.DataFrame] = None) -> "BiasedMatrixFactorization":
        """Fits Biased Matrix Factorization using AdamW on GPU (MPS/CUDA)."""
        targets = train_df["target_id"].to_list()
        ligands = train_df["ligand_smiles"].to_list()
        pIC50 = train_df["pIC50"].to_numpy().astype(np.float32)

        unique_targets = sorted(list(set(targets)))
        unique_ligands = sorted(list(set(ligands)))

        # 1-based indexing so that 0 is unknown/cold
        self.target_to_idx = {t: i + 1 for i, t in enumerate(unique_targets)}
        self.ligand_to_idx = {l: j + 1 for j, l in enumerate(unique_ligands)}
        self.global_mean = float(np.mean(pIC50))

        n_targets = len(unique_targets)
        n_ligands = len(unique_ligands)

        t_idx_arr = np.array([self.target_to_idx[t] for t in targets], dtype=np.int64)
        l_idx_arr = np.array([self.ligand_to_idx[l] for l in ligands], dtype=np.int64)

        dataset = TensorDataset(
            torch.tensor(t_idx_arr, dtype=torch.long),
            torch.tensor(l_idx_arr, dtype=torch.long),
            torch.tensor(pIC50, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model = BiasedMFNet(
            n_targets=n_targets,
            n_ligands=n_ligands,
            latent_dim=self.config.latent_dim,
            global_mean=self.global_mean,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for t_batch, l_batch, y_batch in loader:
                t_batch = t_batch.to(self.device)
                l_batch = l_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(t_batch, l_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_batch)

        self.is_fitted = True
        return self

    def predict_batch(
        self,
        targets: Union[List[str], pl.Series],
        ligands: Union[List[str], pl.Series],
    ) -> np.ndarray:
        """Batch inference on GPU."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        if isinstance(targets, pl.Series):
            targets = targets.to_list()
        if isinstance(ligands, pl.Series):
            ligands = ligands.to_list()

        t_indices = [self.target_to_idx.get(t, 0) for t in targets]
        l_indices = [self.ligand_to_idx.get(l, 0) for l in ligands]

        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t_indices, dtype=torch.long, device=self.device)
            l_tensor = torch.tensor(l_indices, dtype=torch.long, device=self.device)
            preds = self.model(t_tensor, l_tensor).cpu().numpy()

        return preds


# ---------------------------------------------------------------------------
# 2. Two-Tower Hybrid Neural Matrix Factorization
# ---------------------------------------------------------------------------

class TwoTowerHybridNet(nn.Module):
    """Two-Tower Hybrid Architecture with Embedding Dropout."""

    def __init__(
        self,
        n_targets: int,
        n_ligands: int,
        chem_input_dim: int = 2048,
        prot_input_dim: int = 1024,
        latent_dim: int = 64,
        chem_hidden_dim: int = 256,
        prot_hidden_dim: int = 256,
        embedding_dropout: float = 0.25,
        mlp_dropout: float = 0.15,
        global_mean: float = 6.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dropout = embedding_dropout
        self.mu = nn.Parameter(torch.tensor(float(global_mean), dtype=torch.float32), requires_grad=True)

        # ID Embeddings & Biases (index 0 is reserved for cold/unseen)
        self.target_bias = nn.Embedding(n_targets + 1, 1, padding_idx=0)
        self.ligand_bias = nn.Embedding(n_ligands + 1, 1, padding_idx=0)
        self.target_emb = nn.Embedding(n_targets + 1, latent_dim, padding_idx=0)
        self.ligand_emb = nn.Embedding(n_ligands + 1, latent_dim, padding_idx=0)

        # Chemical Feature Tower (MLP on Morgan ECFP4)
        self.chem_tower = nn.Sequential(
            nn.Linear(chem_input_dim, chem_hidden_dim),
            nn.LayerNorm(chem_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(chem_hidden_dim, latent_dim),
        )

        # Protein Sequence Feature Tower (MLP on TF-IDF 3-mers)
        self.prot_tower = nn.Sequential(
            nn.Linear(prot_input_dim, prot_hidden_dim),
            nn.LayerNorm(prot_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(prot_hidden_dim, latent_dim),
        )

        # Initialization
        nn.init.zeros_(self.target_bias.weight)
        nn.init.zeros_(self.ligand_bias.weight)
        nn.init.normal_(self.target_emb.weight, std=0.05)
        nn.init.normal_(self.ligand_emb.weight, std=0.05)

    def forward(
        self,
        t_indices: torch.Tensor,
        l_indices: torch.Tensor,
        chem_features: torch.Tensor,
        prot_features: torch.Tensor,
    ) -> torch.Tensor:
        # Determine masks for embedding dropout
        if self.training and self.embedding_dropout > 0.0:
            mask_t = (
                (torch.rand(t_indices.shape, device=t_indices.device) > self.embedding_dropout)
                & (t_indices > 0)
            ).float().unsqueeze(-1)
            mask_l = (
                (torch.rand(l_indices.shape, device=l_indices.device) > self.embedding_dropout)
                & (l_indices > 0)
            ).float().unsqueeze(-1)
        else:
            mask_t = (t_indices > 0).float().unsqueeze(-1)
            mask_l = (l_indices > 0).float().unsqueeze(-1)

        # Tower representations: ID embedding (masked) + Content projection
        p_t = mask_t * self.target_emb(t_indices) + self.prot_tower(prot_features)
        q_l = mask_l * self.ligand_emb(l_indices) + self.chem_tower(chem_features)

        # Biases (masked if entity is cold or dropped)
        b_t = (mask_t.squeeze(-1) * self.target_bias(t_indices).squeeze(-1))
        b_l = (mask_l.squeeze(-1) * self.ligand_bias(l_indices).squeeze(-1))

        # Latent interaction
        interaction = (p_t * q_l).sum(dim=-1)
        return self.mu + b_t + b_l + interaction


class TwoTowerHybridNeuralMF:
    """Two-Tower Hybrid Neural Matrix Factorization Recommender with hardware acceleration."""

    def __init__(
        self,
        config: Optional[TwoTowerHybridConfig] = None,
        mol_featurizer: Optional[MoleculeFeaturizer] = None,
        prot_featurizer: Optional[ProteinFeaturizer] = None,
    ):
        self.config = config or TwoTowerHybridConfig()
        self.device = torch.device(self.config.device)
        self.mol_featurizer = mol_featurizer or MoleculeFeaturizer()
        self.prot_featurizer = prot_featurizer or ProteinFeaturizer()

        self.model: Optional[TwoTowerHybridNet] = None
        self.target_to_idx: Dict[str, int] = {}
        self.ligand_to_idx: Dict[str, int] = {}
        self.global_mean: float = 6.0
        self.is_fitted: bool = False

    def fit(self, train_df: pl.DataFrame) -> "TwoTowerHybridNeuralMF":
        """Precomputes chem/prot features and trains Two-Tower Neural MF via PyTorch."""
        targets = train_df["target_id"].to_list()
        ligands = train_df["ligand_smiles"].to_list()
        sequences = train_df["sequence"].to_list()
        pIC50 = train_df["pIC50"].to_numpy().astype(np.float32)

        # 1. Fit protein featurizer if needed
        target_seq_dict = {t: s for t, s in zip(targets, sequences)}
        if not self.prot_featurizer.is_fitted:
            self.prot_featurizer.fit(target_seq_dict)

        unique_targets = sorted(list(set(targets)))
        unique_ligands = sorted(list(set(ligands)))

        self.target_to_idx = {t: i + 1 for i, t in enumerate(unique_targets)}
        self.ligand_to_idx = {l: j + 1 for j, l in enumerate(unique_ligands)}
        self.global_mean = float(np.mean(pIC50))

        # 2. Extract features
        print(f"[Two-Tower] Featurizing {len(unique_ligands)} ligands and {len(unique_targets)} targets...")
        chem_features = np.stack([self.mol_featurizer.get_morgan_fp(l) for l in ligands], axis=0)
        prot_features = np.stack([self.prot_featurizer.get_target_embedding(t) for t in targets], axis=0)

        t_indices = np.array([self.target_to_idx[t] for t in targets], dtype=np.int64)
        l_indices = np.array([self.ligand_to_idx[l] for l in ligands], dtype=np.int64)

        dataset = TensorDataset(
            torch.tensor(t_indices, dtype=torch.long),
            torch.tensor(l_indices, dtype=torch.long),
            torch.tensor(chem_features, dtype=torch.float32),
            torch.tensor(prot_features, dtype=torch.float32),
            torch.tensor(pIC50, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        chem_dim = chem_features.shape[1]
        prot_dim = prot_features.shape[1]

        self.model = TwoTowerHybridNet(
            n_targets=len(unique_targets),
            n_ligands=len(unique_ligands),
            chem_input_dim=chem_dim,
            prot_input_dim=prot_dim,
            latent_dim=self.config.latent_dim,
            chem_hidden_dim=self.config.chem_hidden_dim,
            prot_hidden_dim=self.config.prot_hidden_dim,
            embedding_dropout=self.config.embedding_dropout,
            mlp_dropout=self.config.mlp_dropout,
            global_mean=self.global_mean,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for t_b, l_b, chem_b, prot_b, y_b in loader:
                t_b = t_b.to(self.device)
                l_b = l_b.to(self.device)
                chem_b = chem_b.to(self.device)
                prot_b = prot_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                preds = self.model(t_b, l_b, chem_b, prot_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_b)

        self.is_fitted = True
        return self

    def predict_batch(
        self,
        targets: Union[List[str], pl.Series],
        ligands: Union[List[str], pl.Series],
        sequences: Optional[Union[List[str], pl.Series]] = None,
    ) -> np.ndarray:
        """Inference with Two-Tower Hybrid network."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        if isinstance(targets, pl.Series):
            targets = targets.to_list()
        if isinstance(ligands, pl.Series):
            ligands = ligands.to_list()
        seq_list = sequences.to_list() if isinstance(sequences, pl.Series) else sequences

        t_indices = [self.target_to_idx.get(t, 0) for t in targets]
        l_indices = [self.ligand_to_idx.get(l, 0) for l in ligands]

        # Featurize test ligands and targets
        chem_features = np.stack([self.mol_featurizer.get_morgan_fp(l) for l in ligands], axis=0)
        prot_features = np.stack(
            [
                self.prot_featurizer.get_target_embedding(t, fallback_seq=(seq_list[i] if seq_list else None))
                for i, t in enumerate(targets)
            ],
            axis=0,
        )

        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t_indices, dtype=torch.long, device=self.device)
            l_tensor = torch.tensor(l_indices, dtype=torch.long, device=self.device)
            chem_tensor = torch.tensor(chem_features, dtype=torch.float32, device=self.device)
            prot_tensor = torch.tensor(prot_features, dtype=torch.float32, device=self.device)

            preds = self.model(t_tensor, l_tensor, chem_tensor, prot_tensor).cpu().numpy()

        return preds
