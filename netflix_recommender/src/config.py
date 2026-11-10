"""
Global Configuration for the Netflix Prize Recommender System.
Optimized for Apple Silicon (M4 Metal Performance Shaders - MPS).
"""

import os
from pathlib import Path
import torch

# Set temporary directory for Matplotlib cache to prevent permission warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Directory Paths
BASE_DIR = Path("/Users/emilianodesu/Documents/LCD/7Sem/DMII/T4/netflix_recommender").resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

# Ensure target directories exist
for p in [PROCESSED_DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Raw Data Files
MOVIE_TITLES_FILE = RAW_DATA_DIR / "movie_titles.csv"
PROBE_FILE = RAW_DATA_DIR / "probe.txt"
QUALIFYING_FILE = RAW_DATA_DIR / "qualifying.txt"
COMBINED_DATA_FILES = [
    RAW_DATA_DIR / f"combined_data_{i}.txt" for i in range(1, 5)
]

# Processed Data Artifacts
TRAIN_PARQUET = PROCESSED_DATA_DIR / "train.parquet"
PROBE_PARQUET = PROCESSED_DATA_DIR / "probe.parquet"
MOVIES_PARQUET = PROCESSED_DATA_DIR / "movies.parquet"
TRAIN_CSR_FILE = PROCESSED_DATA_DIR / "train_csr.npz"
ID_MAPS_FILE = PROCESSED_DATA_DIR / "id_mappings.json"
METRICS_SUMMARY_FILE = OUTPUTS_DIR / "metrics_summary.json"

# Hardware Acceleration (Apple Silicon Metal MPS)
def get_device() -> torch.device:
    """Returns Apple Silicon MPS device if available, otherwise CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# Global Hyperparameters & Reproducibility
RANDOM_SEED = 42

# Sampling Hyperparameters (Dense Core subset: ~1.2M real ratings)
SAMPLE_CONFIG = {
    "top_movies": 1500,
    "min_user_ratings": 50,
    "num_users": 10000,
    "source_file_index": 0,  # combined_data_1.txt
    "random_seed": RANDOM_SEED,
}

# Baseline Decoupled Regularization Parameters
BASELINE_CONFIG = {
    "lambda_u": 25.0,  # User regularizer
    "lambda_i": 10.0,  # Movie regularizer
    "clip_min": 1.0,
    "clip_max": 5.0,
}

# Item-Item Collaborative Filtering (Residuals + Pearson + Bayesian Shrinkage)
KNN_CONFIG = {
    "k": 40,
    "beta": 25.0,     # Bayesian shrinkage constant for co-ratings
    "gamma": 1.0,    # Normalization shrinkage
    "min_common": 3,  # Minimum co-raters required to compute similarity
    "clip_min": 1.0,
    "clip_max": 5.0,
}

# Biased Matrix Factorization (PyTorch Metal MPS)
BIASED_MF_CONFIG = {
    "n_factors": 32,
    "lr": 0.005,
    "weight_decay": 0.02,
    "epochs": 15,
    "batch_size": 16384,
    "clip_min": 1.0,
    "clip_max": 5.0,
}

# SVD++ with Implicit Feedback (Koren 2008, PyTorch Metal MPS)
SVDPP_CONFIG = {
    "n_factors": 32,
    "lr": 0.005,
    "weight_decay": 0.02,
    "epochs": 15,
    "batch_size": 16384,
    "clip_min": 1.0,
    "clip_max": 5.0,
}

# Content Cold-Start Projection Bridge
COLD_START_CONFIG = {
    "ridge_alpha": 10.0,
    "tfidf_max_features": 1000,
    "ngram_range": (1, 2),
    "simulated_cold_movies": 50,
}

# Ranking and Evaluation
EVALUATION_CONFIG = {
    "top_k_list": [5, 10, 20],
    "relevance_threshold": 4.0,  # 4 and 5 stars considered relevant
    "benchmark_netflix_baseline_rmse": 1.0540,
    "benchmark_cinematch_rmse": 0.9525,
    "benchmark_grand_prize_rmse": 0.8567,
}
