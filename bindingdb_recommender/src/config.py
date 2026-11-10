"""Configuration module for BindingDB Chemogenomic Recommender System.
Centralizes paths, hyperparameters, biochemical thresholds, and runtime options.
"""
from dataclasses import dataclass, field
from pathlib import Path
import torch

# Base project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "BindingDB_BindingDB_Articles.tsv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DATA_DIR = DATA_DIR / "splits"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_PATH = OUTPUTS_DIR / "metrics_summary.json"

# Biochemical constants and thresholds
IC50_CLIP_MIN_NM: float = 1e-3     # 1 picomolar lower bound
IC50_CLIP_MAX_NM: float = 1e7      # 10 millimolar upper bound
ACTIVE_PIC50_THRESHOLD: float = 6.0  # pIC50 >= 6.0 corresponds to IC50 <= 1 uM (active hits)

RANDOM_SEED: int = 42

# Device selection: Apple Silicon Metal (MPS) > CUDA > CPU
if torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
elif torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
else:
    DEFAULT_DEVICE = "cpu"


@dataclass
class BaselineConfig:
    lambda_target: float = 10.0   # L2 shrinkage penalty for target bias
    lambda_ligand: float = 10.0   # L2 shrinkage penalty for ligand bias


@dataclass
class CollaborativeKNNConfig:
    k: int = 25                   # Number of nearest neighbors
    shrinkage_beta: float = 10.0  # Shrinkage towards residual baseline
    min_overlap: int = 2          # Minimum co-tested entities
    mode: str = "target"          # "target" (target-target similarity) or "ligand"


@dataclass
class ContentChemogenomicConfig:
    k: int = 25                   # Number of nearest neighbors
    shrinkage_beta: float = 5.0   # Shrinkage factor towards target/ligand mean
    morgan_radius: int = 2        # Radius 2 gives ECFP4
    morgan_nbits: int = 2048      # ECFP4 bit length
    protein_kmer_size: int = 3    # 3-mer amino acid sequences
    protein_tfidf_max_features: int = 1024


@dataclass
class LatentFactorsGPUConfig:
    latent_dim: int = 64
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    epochs: int = 25
    batch_size: int = 512
    device: str = DEFAULT_DEVICE


@dataclass
class TwoTowerHybridConfig:
    latent_dim: int = 64
    chem_hidden_dim: int = 256
    prot_hidden_dim: int = 256
    embedding_dropout: float = 0.25   # Dropout probability for entity ID embeddings
    mlp_dropout: float = 0.15
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 512
    device: str = DEFAULT_DEVICE


@dataclass
class PipelineConfig:
    raw_tsv_path: Path = RAW_DATA_PATH
    processed_dir: Path = PROCESSED_DATA_DIR
    splits_dir: Path = SPLITS_DATA_DIR
    figures_dir: FIGURES_DIR = FIGURES_DIR
    metrics_path: Path = METRICS_PATH
    
    # Dataset filtering and sampling
    sample_mode: bool = False
    sample_top_targets: int = 40       # Used in --sample mode: top targets by assays
    min_ligands_per_target: int = 5     # Minimum assays per target in full mode
    
    # Splits proportions
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    
    # Model subconfigs
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    collab_knn: CollaborativeKNNConfig = field(default_factory=CollaborativeKNNConfig)
    content: ContentChemogenomicConfig = field(default_factory=ContentChemogenomicConfig)
    biased_mf: LatentFactorsGPUConfig = field(default_factory=LatentFactorsGPUConfig)
    two_tower: TwoTowerHybridConfig = field(default_factory=TwoTowerHybridConfig)
