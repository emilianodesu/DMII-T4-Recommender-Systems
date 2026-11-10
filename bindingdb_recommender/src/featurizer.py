"""Featurization module for molecules and protein targets.
Provides:
1) Molecular Featurization:
   - Morgan Circular Fingerprints (ECFP4, 2048 bits)
   - MACCS Keys (166 bits)
   - Continuous Physicochemical Lipinski Descriptors (MW, LogP, HBD, HBA, TPSA, Rotatable Bonds)
2) Protein Featurization:
   - Amino acid 3-mers vectorized via TF-IDF with L2 normalization
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.feature_extraction.text import TfidfVectorizer


class MoleculeFeaturizer:
    """Computes chemoinformatic representations for small molecules using RDKit."""

    def __init__(self, morgan_radius: int = 2, morgan_nbits: int = 2048):
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self._morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=morgan_radius,
            fpSize=morgan_nbits,
        )
        self._cache_morgan: Dict[str, np.ndarray] = {}
        self._cache_maccs: Dict[str, np.ndarray] = {}
        self._cache_lipinski: Dict[str, np.ndarray] = {}

    def get_morgan_fp(self, smiles: str) -> np.ndarray:
        """Computes Morgan ECFP4 fingerprint as a binary/float vector of length morgan_nbits."""
        if smiles in self._cache_morgan:
            return self._cache_morgan[smiles]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = self._morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            else:
                fp = np.zeros(self.morgan_nbits, dtype=np.float32)
        except Exception:
            fp = np.zeros(self.morgan_nbits, dtype=np.float32)

        self._cache_morgan[smiles] = fp
        return fp

    def get_maccs_keys(self, smiles: str) -> np.ndarray:
        """Computes 167-bit MACCS structural key fingerprint."""
        if smiles in self._cache_maccs:
            return self._cache_maccs[smiles]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                maccs = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros((len(maccs),), dtype=np.float32)
                ConvertToNumpyArray(maccs, arr)
            else:
                arr = np.zeros(167, dtype=np.float32)
        except Exception:
            arr = np.zeros(167, dtype=np.float32)

        self._cache_maccs[smiles] = arr
        return arr

    def get_lipinski_descriptors(self, smiles: str) -> np.ndarray:
        """Computes continuous physicochemical descriptors:
        [MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds].
        """
        if smiles in self._cache_lipinski:
            return self._cache_lipinski[smiles]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc = np.array([
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    float(Descriptors.NumHDonors(mol)),
                    float(Descriptors.NumHAcceptors(mol)),
                    Descriptors.TPSA(mol),
                    float(Descriptors.NumRotatableBonds(mol)),
                ], dtype=np.float32)
            else:
                desc = np.zeros(6, dtype=np.float32)
        except Exception:
            desc = np.zeros(6, dtype=np.float32)

        self._cache_lipinski[smiles] = desc
        return desc

    def featurize_smiles_list(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precomputes Morgan, MACCS, and Lipinski matrices for a list of SMILES."""
        morgan_list = [self.get_morgan_fp(s) for s in smiles_list]
        maccs_list = [self.get_maccs_keys(s) for s in smiles_list]
        lipinski_list = [self.get_lipinski_descriptors(s) for s in smiles_list]

        return (
            np.stack(morgan_list, axis=0),
            np.stack(maccs_list, axis=0),
            np.stack(lipinski_list, axis=0),
        )


class ProteinFeaturizer:
    """Extracts overlapping amino acid k-mers from protein primary sequences
    and encodes them as L2-normalized TF-IDF vector representations.
    """

    def __init__(self, kmer_size: int = 3, max_features: int = 1024):
        self.kmer_size = kmer_size
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            token_pattern=r"\b\w+\b",
            max_features=max_features,
            norm="l2",
        )
        self.is_fitted = False
        self._target_seq_map: Dict[str, str] = {}
        self._target_vec_map: Dict[str, np.ndarray] = {}

    def _sequence_to_kmers(self, seq: str) -> str:
        """Converts an amino acid sequence to space-separated overlapping k-mers."""
        if not seq or len(seq) < self.kmer_size:
            return "UNK"
        clean_seq = "".join([c for c in seq.upper() if c.isalpha()])
        if len(clean_seq) < self.kmer_size:
            return "UNK"
        kmers = [
            clean_seq[i : i + self.kmer_size]
            for i in range(len(clean_seq) - self.kmer_size + 1)
        ]
        return " ".join(kmers)

    def fit(self, target_to_seq: Dict[str, str]) -> "ProteinFeaturizer":
        """Fits TF-IDF model on a dictionary of {target_id: amino_acid_sequence}."""
        self._target_seq_map = dict(target_to_seq)
        docs = [self._sequence_to_kmers(s) for s in target_to_seq.values()]
        self.vectorizer.fit(docs)
        self.is_fitted = True

        # Precompute target embeddings
        X = self.vectorizer.transform(docs).toarray().astype(np.float32)
        for t_id, vec in zip(target_to_seq.keys(), X):
            self._target_vec_map[t_id] = vec
        return self

    def transform_sequence(self, seq: str) -> np.ndarray:
        """Transforms a single protein sequence into a TF-IDF vector."""
        if not self.is_fitted:
            raise RuntimeError("ProteinFeaturizer must be fitted before transforming sequences.")
        doc = self._sequence_to_kmers(seq)
        vec = self.vectorizer.transform([doc]).toarray()[0].astype(np.float32)
        return vec

    def get_target_embedding(self, target_id: str, fallback_seq: Optional[str] = None) -> np.ndarray:
        """Returns the precomputed embedding for target_id, or computes it on the fly if unseen."""
        if target_id in self._target_vec_map:
            return self._target_vec_map[target_id]
        if fallback_seq is not None:
            vec = self.transform_sequence(fallback_seq)
            self._target_vec_map[target_id] = vec
            return vec
        # Fallback to zero vector if completely unknown
        return np.zeros(self.max_features, dtype=np.float32)

    def get_all_target_embeddings(self) -> Dict[str, np.ndarray]:
        """Returns target embedding dictionary."""
        return self._target_vec_map
