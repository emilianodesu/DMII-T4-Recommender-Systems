"""
High-Resolution Visualization Module for Netflix Prize Recommender System.
Generates publication-quality figures saved into outputs/figures/:
1. eda_ratings_distribution.png
2. rmse_model_comparison.png
3. training_curves_gpu.png
4. ensemble_weights.png
5. ranking_ndcg_recall_k.png
6. cold_start_projection.png
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from src.config import FIGURES_DIR, EVALUATION_CONFIG

logger = logging.getLogger(__name__)

# Configure plot aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_eda_ratings_distribution(
    df_train: pl.DataFrame,
    df_probe: pl.DataFrame,
    save_path: Path = FIGURES_DIR / "eda_ratings_distribution.png",
):
    """Generates 3-panel EDA plot: rating star frequencies, long-tail movie counts, user activity."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # 1. Rating Stars Distribution
    counts = df_train["rating"].value_counts().sort("rating")
    ratings = counts["rating"].to_numpy()
    freqs = counts["count"].to_numpy()
    pcts = (freqs / freqs.sum()) * 100

    bars = axes[0].bar(ratings, freqs, color="#E50914", edgecolor="#831010", alpha=0.85, width=0.6)
    axes[0].set_title("Netflix Prize Rating Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Star Rating", fontsize=11)
    axes[0].set_ylabel("Total Count", fontsize=11)
    axes[0].set_xticks([1, 2, 3, 4, 5])
    for bar, pct in zip(bars, pcts):
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + (freqs.max() * 0.015), f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Movie Rating Distribution (Long Tail)
    movie_counts = df_train.group_by("movie_idx").len().sort("len", descending=True)["len"].to_numpy()
    axes[1].plot(np.arange(len(movie_counts)), movie_counts, color="#221F1F", lw=2)
    axes[1].fill_between(np.arange(len(movie_counts)), movie_counts, color="#E50914", alpha=0.2)
    axes[1].set_title("Catalog Popularity (Long-Tail)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Movie Rank (Popularity)", fontsize=11)
    axes[1].set_ylabel("Ratings per Movie", fontsize=11)

    # 3. User Rating Activity
    user_counts = df_train.group_by("user_idx").len()["len"].to_numpy()
    axes[2].hist(user_counts, bins=40, color="#564D4D", edgecolor="black", alpha=0.8)
    axes[2].set_title("User Engagement Distribution", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Ratings per User", fontsize=11)
    axes[2].set_ylabel("Number of Users", fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved EDA distribution figure to {save_path}")


def plot_rmse_comparison(
    metrics: Dict[str, Dict[str, Any]],
    save_path: Path = FIGURES_DIR / "rmse_model_comparison.png",
):
    """Bar chart comparing official Probe RMSE across all models with historical benchmarks."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    model_names = [m for m in ["Baseline", "Item_kNN", "Biased_MF", "SVDPP", "Ensemble"] if m in metrics]
    rmses = [metrics[m]["RMSE"] for m in model_names]

    colors = ["#7f7f7f", "#3b528b", "#21918c", "#5ec962", "#E50914"]
    bars = ax.bar(model_names, rmses, color=colors[:len(model_names)], width=0.55, edgecolor="black", alpha=0.9)

    # Historical Netflix Benchmarks
    cinematch = EVALUATION_CONFIG["benchmark_cinematch_rmse"]
    netflix_base = EVALUATION_CONFIG["benchmark_netflix_baseline_rmse"]
    grand_prize = EVALUATION_CONFIG["benchmark_grand_prize_rmse"]

    ax.axhline(cinematch, color="orange", linestyle="--", linewidth=1.5, label=f"Cinematch ({cinematch:.4f})")
    ax.axhline(netflix_base, color="red", linestyle=":", linewidth=1.5, label=f"Netflix Global Baseline ({netflix_base:.4f})")
    ax.axhline(grand_prize, color="green", linestyle="-.", linewidth=1.5, label=f"10% Grand Prize Target ({grand_prize:.4f})")

    ax.set_ylim(0.75, max(rmses + [netflix_base]) * 1.05)
    ax.set_ylabel("Official Probe RMSE (Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Progression on Official Netflix Probe Set", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)

    for bar, rmse in zip(bars, rmses):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f"{rmse:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved RMSE comparison figure to {save_path}")


def plot_training_curves(
    bmf_history: List[Dict[str, float]],
    svd_history: List[Dict[str, float]],
    save_path: Path = FIGURES_DIR / "training_curves_gpu.png",
):
    """Training loss and validation RMSE curves over epochs on Apple Silicon GPU MPS."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Loss curve
    bmf_epochs = [h["epoch"] for h in bmf_history]
    bmf_loss = [h["train_loss"] for h in bmf_history]
    svd_epochs = [h["epoch"] for h in svd_history]
    svd_loss = [h["train_loss"] for h in svd_history]

    ax1.plot(bmf_epochs, bmf_loss, marker="o", color="#21918c", lw=2, label="Biased MF (MPS)")
    ax1.plot(svd_epochs, svd_loss, marker="s", color="#5ec962", lw=2, label="SVD++ (MPS)")
    ax1.set_title("Training Loss (MSE) per Epoch", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Squared Error", fontsize=11)
    ax1.legend(frameon=True)

    # Probe RMSE curve
    bmf_rmse = [h["val_rmse"] for h in bmf_history if h.get("val_rmse") is not None]
    svd_rmse = [h["val_rmse"] for h in svd_history if h.get("val_rmse") is not None]

    if bmf_rmse and svd_rmse:
        ax2.plot(bmf_epochs[:len(bmf_rmse)], bmf_rmse, marker="o", color="#21918c", lw=2, label="Biased MF Val RMSE")
        ax2.plot(svd_epochs[:len(svd_rmse)], svd_rmse, marker="s", color="#5ec962", lw=2, label="SVD++ Val RMSE")
        ax2.set_title("Validation RMSE on Probe Set", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("RMSE", fontsize=11)
        ax2.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved GPU training curves figure to {save_path}")


def plot_ensemble_weights(
    weights_dict: Dict[str, float],
    save_path: Path = FIGURES_DIR / "ensemble_weights.png",
):
    """Horizontal bar chart showing stacking blending weights."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    models = list(weights_dict.keys())
    w_vals = list(weights_dict.values())
    colors = sns.color_palette("viridis", len(models))

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, w_vals, color=colors, edgecolor="black", alpha=0.85, height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, fontweight="bold")
    ax.set_xlabel("Blending Weight (sum = 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Optimal Ensemble Stacking Weights (SLSQP Quadratic Program)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(w_vals) * 1.25 if w_vals else 1.0)

    for bar, val in zip(bars, w_vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f} ({val*100:.1f}%)", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ensemble weights figure to {save_path}")


def plot_ranking_metrics(
    metrics: Dict[str, Dict[str, Any]],
    save_path: Path = FIGURES_DIR / "ranking_ndcg_recall_k.png",
):
    """Bar plot comparing NDCG@K, Recall@K, and Precision@K across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    models = [m for m in ["Baseline", "Item_kNN", "Biased_MF", "SVDPP", "Ensemble"] if m in metrics]
    k_vals = [5, 10, 20]

    # NDCG@K
    x = np.arange(len(models))
    width = 0.25

    for idx, k in enumerate(k_vals):
        ndcgs = [metrics[m].get(f"NDCG@{k}", 0.0) for m in models]
        ax1.bar(x + (idx - 1) * width, ndcgs, width, label=f"NDCG@{k}")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylabel("NDCG Score", fontsize=11)
    ax1.set_title("Ranking Quality: NDCG@K across Models", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)

    # Recall@K
    for idx, k in enumerate(k_vals):
        recalls = [metrics[m].get(f"Recall@{k}", 0.0) for m in models]
        ax2.bar(x + (idx - 1) * width, recalls, width, label=f"Recall@{k}")

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylabel("Recall Score", fontsize=11)
    ax2.set_title("Recommendation Coverage: Recall@K across Models", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ranking metrics figure to {save_path}")


def plot_cold_start_projection(
    latent_factors: np.ndarray,
    projected_factors: np.ndarray,
    save_path: Path = FIGURES_DIR / "cold_start_projection.png",
):
    """PCA 2D projection comparing learned SVD++ latent item factors with content-projected cold-start items."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    pca = PCA(n_components=2, random_state=42)
    n_train = len(latent_factors)
    combined = np.vstack([latent_factors, projected_factors])
    transformed = pca.fit_transform(combined)

    train_2d = transformed[:n_train]
    cold_2d = transformed[n_train:]

    ax.scatter(train_2d[:, 0], train_2d[:, 1], c="#3b528b", alpha=0.4, s=25, label="Train Items (SVD++ factors)")
    ax.scatter(cold_2d[:, 0], cold_2d[:, 1], c="#E50914", marker="*", s=90, edgecolors="black", label="Cold-Start Items (Ridge Bridge Projection)")

    ax.set_title(f"Content Cold-Start Bridge: Alignment in Latent Space (PCA Expl. Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("First Principal Component", fontsize=11)
    ax.set_ylabel("Second Principal Component", fontsize=11)
    ax.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cold-start projection figure to {save_path}")
