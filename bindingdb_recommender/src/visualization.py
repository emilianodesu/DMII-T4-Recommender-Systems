"""Diagnostic Visualization Module for BindingDB Chemogenomic Recommender.
Generates publication-quality diagnostic plots in outputs/figures/:
1) figure1_pIC50_distribution.png: Curation distribution and active threshold (pIC50 >= 6.0).
2) figure2_model_benchmark_comparison.png: Multi-split performance comparison (RMSE, Spearman rho).
3) figure3_parity_predicted_vs_true.png: Experimental vs Predicted pIC50 correlation plots.
4) figure4_target_similarity_heatmap.png: Chemogenomic target similarity structure.
"""
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Clean, professional style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_pic50_distribution(df: pl.DataFrame, output_path: Path) -> None:
    """Plots distribution of experimental pIC50 values and the active threshold."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pic50 = df["pIC50"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(
        pic50,
        bins=45,
        kde=True,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.85,
        ax=ax,
    )

    ax.axvline(
        6.0,
        color="#d9534f",
        linestyle="--",
        linewidth=2,
        label="Active Hit Threshold ($pIC_{50} \\geq 6.0$, $IC_{50} \\leq 1\\mu M$)",
    )

    mean_val = np.mean(pic50)
    median_val = np.median(pic50)
    ax.axvline(
        mean_val,
        color="#5cb85c",
        linestyle=":",
        linewidth=1.8,
        label=f"Mean: {mean_val:.2f}",
    )

    ax.set_title("BindingDB Experimental Affinity Distribution ($pIC_{50}$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experimental $pIC_{50} = 9.0 - \\log_{10}(IC_{50}\\;[nM])$", fontsize=11)
    ax.set_ylabel("Pair Count (Unique Target-Ligand)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved {output_path}")


def plot_benchmark_comparison(metrics_summary: Dict[str, Dict[str, Dict[str, float]]], output_path: Path) -> None:
    """Plots grouped bar comparison of models across Random, Scaffold, and Target splits."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    splits = [s for s in ["random_split", "scaffold_split", "target_split"] if s in metrics_summary]
    if not splits:
        return

    # Extract model names
    first_split = splits[0]
    models = list(metrics_summary[first_split].keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    x = np.arange(len(models))
    width = 0.25

    colors = ["#2b5c8f", "#f0ad4e", "#d9534f"]

    for i, split in enumerate(splits):
        rmse_vals = [metrics_summary[split][m].get("rmse", 0.0) for m in models]
        rho_vals = [metrics_summary[split][m].get("spearman_rho", 0.0) for m in models]

        offset = (i - len(splits) / 2 + 0.5) * width
        label = split.replace("_", " ").title()
        ax1.bar(x + offset, rmse_vals, width, label=label, color=colors[i % len(colors)], alpha=0.9)
        ax2.bar(x + offset, rho_vals, width, label=label, color=colors[i % len(colors)], alpha=0.9)

    ax1.set_title("RMSE by Model and Split (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax1.axhline(1.0, color="#d9534f", linestyle="--", linewidth=1.2, label="Threshold RMSE < 1.0")
    ax1.legend(frameon=True)

    ax2.set_title("Spearman Rank Correlation (Higher is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Spearman $\\rho$", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax2.legend(frameon=True)

    plt.suptitle("Tema 4 Chemogenomic Recommender Benchmark", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved {output_path}")


def plot_parity_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split_name: str, output_path: Path) -> None:
    """Plots parity plot (Predicted vs Experimental pIC50) with density contours."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)

    # Subsample if large for clean visualization
    if len(y_true) > 5000:
        idx = np.random.choice(len(y_true), size=5000, replace=False)
        yt_plot = y_true[idx]
        yp_plot = y_pred[idx]
    else:
        yt_plot = y_true
        yp_plot = y_pred

    ax.scatter(yt_plot, yp_plot, alpha=0.35, edgecolors="none", s=18, color="#2b5c8f")

    # Parity reference line
    lims = [
        min(np.min(yt_plot), np.min(yp_plot)) - 0.2,
        max(np.max(yt_plot), np.max(yp_plot)) + 0.2,
    ]
    ax.plot(lims, lims, "--", color="#d9534f", linewidth=1.5, label="Ideal Parity ($y = \\hat{y}$)")

    # Active threshold line
    ax.axvline(6.0, color="#5cb85c", linestyle=":", alpha=0.8, label="Active ($pIC_{50} = 6.0$)")
    ax.axhline(6.0, color="#5cb85c", linestyle=":", alpha=0.8)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (np.sum((y_true - y_pred) ** 2) / (ss_tot + 1e-9))

    ax.set_title(f"{model_name} on {split_name.replace('_', ' ').title()}\nRMSE: {rmse:.3f} | $R^2$: {r2:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Experimental $pIC_{50}$", fontsize=10)
    ax.set_ylabel("Predicted $pIC_{50}$", fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved {output_path}")


def plot_target_similarity_matrix(sim_matrix: np.ndarray, target_labels: List[str], output_path: Path) -> None:
    """Plots heatmap of target-target collaborative similarity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # If too many targets, take top 25 for clear visualization
    n_display = min(25, len(target_labels))
    sub_sim = sim_matrix[:n_display, :n_display]
    sub_labels = target_labels[:n_display]

    sns.heatmap(
        sub_sim,
        xticklabels=sub_labels,
        yticklabels=sub_labels,
        cmap="Blues",
        annot=False,
        square=True,
        cbar_kws={"label": "Residual Cosine Similarity $s(t, t')$"},
        ax=ax,
    )

    ax.set_title("Target-Target Collaborative Similarity Matrix", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved {output_path}")
