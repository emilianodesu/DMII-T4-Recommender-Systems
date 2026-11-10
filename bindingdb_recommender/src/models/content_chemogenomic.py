"""Content-Based Chemogenomic Recommender Model.
Uses Tanimoto (Jaccard) similarity on Morgan ECFP4 fingerprints for ligands,
and Cosine similarity on amino acid 3-mers TF-IDF embeddings for protein targets.
Specifically engineered to accurately predict binding affinities in Cold-Start regimes:
- Cold-Ligand: unseen molecules tested against known targets.
- Cold-Target: unseen proteins tested against known molecules.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl

from src.config import ContentChemogenomicConfig
from src.featurizer import MoleculeFeaturizer, ProteinFeaturizer
from src.models.baseline import RegularizedBaseline


def compute_tanimoto_vectorized(fp_query: np.ndarray, fp_database: np.ndarray) -> np.ndarray:
    """Vectorized Tanimoto similarity between a single query fingerprint (2048,)
    and a database matrix (N, 2048).
    """
    # Dot product: intersection
    intersection = fp_database @ fp_query
    query_sum = np.sum(fp_query)
    db_sum = np.sum(fp_database, axis=1)
    union = db_sum + query_sum - intersection
    tanimoto = np.where(union > 0, intersection / np.maximum(union, 1e-7), 0.0)
    return tanimoto.astype(np.float32)


def compute_cosine_vectorized(vec_query: np.ndarray, vec_database: np.ndarray) -> np.ndarray:
    """Cosine similarity between query (D,) and database (N, D).
    Assumes vectors are L2 normalized or normalizes on the fly.
    """
    norm_q = np.linalg.norm(vec_query)
    if norm_q < 1e-7:
        return np.zeros(len(vec_database), dtype=np.float32)
    q_unit = vec_query / norm_q

    norms_db = np.linalg.norm(vec_database, axis=1)
    norms_db = np.where(norms_db < 1e-7, 1.0, norms_db)
    sims = (vec_database @ q_unit) / norms_db
    return np.clip(sims, -1.0, 1.0).astype(np.float32)


class ContentChemogenomicKNN:
    """Content-based Chemogenomic Nearest Neighbor Recommender."""

    def __init__(
        self,
        config: Optional[ContentChemogenomicConfig] = None,
        baseline: Optional[RegularizedBaseline] = None,
        mol_featurizer: Optional[MoleculeFeaturizer] = None,
        prot_featurizer: Optional[ProteinFeaturizer] = None,
    ):
        self.config = config or ContentChemogenomicConfig()
        self.baseline = baseline or RegularizedBaseline()
        self.mol_featurizer = mol_featurizer or MoleculeFeaturizer(
            morgan_radius=self.config.morgan_radius,
            morgan_nbits=self.config.morgan_nbits,
        )
        self.prot_featurizer = prot_featurizer or ProteinFeaturizer(
            kmer_size=self.config.protein_kmer_size,
            max_features=self.config.protein_tfidf_max_features,
        )

        # Index data structures:
        # Per target: list of ligand SMILES, their Morgan FPs, and observed pIC50
        self.target_profiles: Dict[str, Dict[str, Union[np.ndarray, List[str]]]] = {}
        # Per ligand: list of target IDs, their protein embeddings, and observed pIC50
        self.ligand_profiles: Dict[str, Dict[str, Union[np.ndarray, List[str]]]] = {}

        self.is_fitted: bool = False

    def fit(self, train_df: pl.DataFrame) -> "ContentChemogenomicKNN":
        """Fits baseline, featurizes training entities, and builds chemogenomic content profiles."""
        self.baseline.fit(train_df)

        targets = train_df["target_id"].to_list()
        ligands = train_df["ligand_smiles"].to_list()
        sequences = train_df["sequence"].to_list()
        pIC50 = train_df["pIC50"].to_numpy().astype(np.float32)

        # 1. Fit protein featurizer on all unique target sequences
        target_seq_dict = {}
        for t, s in zip(targets, sequences):
            if t not in target_seq_dict:
                target_seq_dict[t] = s
        self.prot_featurizer.fit(target_seq_dict)

        # 2. Featurize unique ligands
        unique_ligands = sorted(list(set(ligands)))
        print(f"[Content-kNN] Precomputing Morgan fingerprints for {len(unique_ligands)} training ligands...")
        for l in unique_ligands:
            self.mol_featurizer.get_morgan_fp(l)

        # 3. Build target-centric profiles
        # Group by target
        target_groups: Dict[str, List[Tuple[str, float]]] = {}
        ligand_groups: Dict[str, List[Tuple[str, float]]] = {}
        for t, l, y in zip(targets, ligands, pIC50):
            target_groups.setdefault(t, []).append((l, y))
            ligand_groups.setdefault(l, []).append((t, y))

        self.target_profiles = {}
        for t, items in target_groups.items():
            t_ligands = [it[0] for it in items]
            t_y = np.array([it[1] for it in items], dtype=np.float32)
            t_fps = np.stack([self.mol_featurizer.get_morgan_fp(l) for l in t_ligands], axis=0)
            self.target_profiles[t] = {
                "ligands": t_ligands,
                "fps": t_fps,
                "y": t_y,
            }

        # 4. Build ligand-centric profiles
        self.ligand_profiles = {}
        for l, items in ligand_groups.items():
            l_targets = [it[0] for it in items]
            l_y = np.array([it[1] for it in items], dtype=np.float32)
            l_embs = np.stack([self.prot_featurizer.get_target_embedding(t) for t in l_targets], axis=0)
            self.ligand_profiles[l] = {
                "targets": l_targets,
                "embs": l_embs,
                "y": l_y,
            }

        self.is_fitted = True
        return self

    def predict_pair(
        self,
        target_id: str,
        ligand_smiles: str,
        target_seq: Optional[str] = None,
    ) -> float:
        """Predicts affinity for a single pair using content similarity."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        b_tl = self.baseline.predict_pair(target_id, ligand_smiles)
        bt = self.baseline.global_mean + self.baseline.b_target.get(target_id, 0.0)
        bl = self.baseline.global_mean + self.baseline.b_ligand.get(ligand_smiles, 0.0)

        has_target_profile = target_id in self.target_profiles
        has_ligand_profile = ligand_smiles in self.ligand_profiles

        # Case 1: Target is known (tested before), ligand may be cold or warm
        if has_target_profile:
            t_prof = self.target_profiles[target_id]
            query_fp = self.mol_featurizer.get_morgan_fp(ligand_smiles)
            tanimoto_sims = compute_tanimoto_vectorized(query_fp, t_prof["fps"])

            # Exclude exact duplicate if present during evaluation
            # Keep top-k with similarity > 0
            pos_mask = tanimoto_sims > 0.0
            if np.any(pos_mask):
                valid_sims = tanimoto_sims[pos_mask]
                valid_y = t_prof["y"][pos_mask]

                k = self.config.k
                if len(valid_sims) > k:
                    top_idx = np.argpartition(valid_sims, -k)[-k:]
                    valid_sims = valid_sims[top_idx]
                    valid_y = valid_y[top_idx]

                # Weighted average with shrinkage towards target mean bt
                # delta = sum(s * (y_i - bt)) / (sum(s) + beta)
                num = np.sum(valid_sims * (valid_y - bt))
                denom = np.sum(valid_sims) + self.config.shrinkage_beta
                pred = bt + (num / denom)
                return float(pred)

        # Case 2: Target is cold (unseen), but ligand is known
        if has_ligand_profile:
            l_prof = self.ligand_profiles[ligand_smiles]
            query_emb = self.prot_featurizer.get_target_embedding(target_id, fallback_seq=target_seq)
            cos_sims = compute_cosine_vectorized(query_emb, l_prof["embs"])

            pos_mask = cos_sims > 0.0
            if np.any(pos_mask):
                valid_sims = cos_sims[pos_mask]
                valid_y = l_prof["y"][pos_mask]

                k = self.config.k
                if len(valid_sims) > k:
                    top_idx = np.argpartition(valid_sims, -k)[-k:]
                    valid_sims = valid_sims[top_idx]
                    valid_y = valid_y[top_idx]

                num = np.sum(valid_sims * (valid_y - bl))
                denom = np.sum(valid_sims) + self.config.shrinkage_beta
                pred = bl + (num / denom)
                return float(pred)

        # Fallback to baseline
        return float(b_tl)

    def predict_batch(
        self,
        targets: Union[List[str], pl.Series],
        ligands: Union[List[str], pl.Series],
        sequences: Optional[Union[List[str], pl.Series]] = None,
    ) -> np.ndarray:
        """Batch prediction using content chemogenomics."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        if isinstance(targets, pl.Series):
            targets = targets.to_list()
        if isinstance(ligands, pl.Series):
            ligands = ligands.to_list()
        seq_list = sequences.to_list() if isinstance(sequences, pl.Series) else sequences

        preds = np.empty(len(targets), dtype=np.float32)
        for i, (t, l) in enumerate(zip(targets, ligands)):
            seq = seq_list[i] if seq_list is not None else None
            preds[i] = self.predict_pair(t, l, target_seq=seq)
        return preds
