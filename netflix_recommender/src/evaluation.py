"""
Evaluation Suite for Netflix Prize Recommender System.
Computes:
  - Official Netflix Prize RMSE
  - Mean Absolute Error (MAE)
  - Precision@K, Recall@K, NDCG@K
  - Catalog Coverage@K
  - Novelty@K (information-theoretic self-information)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import polars as pl

from src.config import EVALUATION_CONFIG

logger = logging.getLogger(__name__)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes official Root Mean Squared Error (RMSE)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error (MAE)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def dcg_at_k(binary_relevance: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at rank K."""
    r = np.asarray(binary_relevance, dtype=np.float64)[:k]
    if r.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return float(np.sum((2.0 ** r - 1.0) / discounts))


def ndcg_at_k(binary_relevance: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    actual_dcg = dcg_at_k(binary_relevance, k)
    ideal_relevance = np.sort(binary_relevance)[::-1]
    best_dcg = dcg_at_k(ideal_relevance, k)
    if best_dcg <= 0.0:
        return 0.0
    return float(actual_dcg / best_dcg)


def evaluate_ranking_and_diversity(
    df_probe: pl.DataFrame,
    predictions: np.ndarray,
    top_k_list: List[int] = EVALUATION_CONFIG["top_k_list"],
    relevance_threshold: float = EVALUATION_CONFIG["relevance_threshold"],
    item_popularity_counts: Optional[Dict[int, int]] = None,
    total_catalog_items: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes ranking metrics (Precision@K, Recall@K, NDCG@K) and diversity/novelty metrics
    grouped per user in the probe evaluation split.
    """
    df_eval = df_probe.with_columns(pl.Series("pred", predictions))

    # Pre-calculate item probabilities for novelty: p(i) = count(i) / total_counts
    item_probs = {}
    if item_popularity_counts:
        total_counts = max(1, sum(item_popularity_counts.values()))
        item_probs = {mid: c / total_counts for mid, c in item_popularity_counts.items()}

    # Group probe rows by user
    user_data = defaultdict(list)
    for row in df_eval.iter_rows(named=True):
        u = row["user_idx"]
        user_data[u].append((row["movie_idx"], row["rating"], row["pred"]))

    metrics_by_k = {k: {"precision": [], "recall": [], "ndcg": [], "recommended_items": set(), "novelties": []} for k in top_k_list}

    for u, items in user_data.items():
        # Items sorted descending by predicted rating
        items_sorted = sorted(items, key=lambda x: x[2], reverse=True)
        actual_ratings = np.array([x[1] for x in items_sorted])
        movie_indices = [x[0] for x in items_sorted]

        # Binary relevance: rating >= threshold (e.g. 4.0)
        bin_rel = (actual_ratings >= relevance_threshold).astype(np.float64)
        total_relevant = int(np.sum(bin_rel))

        if total_relevant == 0:
            continue

        for k in top_k_list:
            top_k_rel = bin_rel[:k]
            rel_in_top_k = int(np.sum(top_k_rel))

            precision = rel_in_top_k / k
            recall = rel_in_top_k / total_relevant
            ndcg = ndcg_at_k(bin_rel, k)

            metrics_by_k[k]["precision"].append(precision)
            metrics_by_k[k]["recall"].append(recall)
            metrics_by_k[k]["ndcg"].append(ndcg)

            # Diversity tracking
            top_k_movies = movie_indices[:k]
            metrics_by_k[k]["recommended_items"].update(top_k_movies)

            # Novelty: -log2(p(i))
            if item_probs:
                nov_vals = [-np.log2(max(item_probs.get(m, 1e-6), 1e-6)) for m in top_k_movies]
                metrics_by_k[k]["novelties"].append(np.mean(nov_vals))

    summary = {}
    catalog_size = total_catalog_items or (int(df_probe["movie_idx"].max()) + 1)

    for k in top_k_list:
        p_mean = float(np.mean(metrics_by_k[k]["precision"])) if metrics_by_k[k]["precision"] else 0.0
        r_mean = float(np.mean(metrics_by_k[k]["recall"])) if metrics_by_k[k]["recall"] else 0.0
        n_mean = float(np.mean(metrics_by_k[k]["ndcg"])) if metrics_by_k[k]["ndcg"] else 0.0
        coverage = float(len(metrics_by_k[k]["recommended_items"]) / catalog_size)
        novelty = float(np.mean(metrics_by_k[k]["novelties"])) if metrics_by_k[k]["novelties"] else 0.0

        summary[f"Precision@{k}"] = round(p_mean, 4)
        summary[f"Recall@{k}"] = round(r_mean, 4)
        summary[f"NDCG@{k}"] = round(n_mean, 4)
        summary[f"Coverage@{k}"] = round(coverage, 4)
        summary[f"Novelty@{k}"] = round(novelty, 4)

    return summary


def evaluate_all_models(
    models_dict: Dict[str, Any],
    df_probe: pl.DataFrame,
    df_train: Optional[pl.DataFrame] = None,
    ensemble_preds: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs full evaluation across all candidate models and the ensemble.
    Returns nested dictionary with RMSE, MAE, and top-K ranking/diversity metrics.
    """
    logger.info("Evaluating all recommendation models on official probe set...")
    u_arr = df_probe["user_idx"].to_numpy()
    m_arr = df_probe["movie_idx"].to_numpy()
    y_true = df_probe["rating"].to_numpy()

    # Precompute item popularity counts from train
    pop_counts = {}
    catalog_size = int(df_probe["movie_idx"].max()) + 1
    if df_train is not None:
        movie_counts = df_train.group_by("movie_idx").len()
        for row in movie_counts.iter_rows(named=True):
            pop_counts[row["movie_idx"]] = row["len"]
        catalog_size = max(catalog_size, int(df_train["movie_idx"].max()) + 1)

    results = {}

    for name, model in models_dict.items():
        preds = model.predict_batch(u_arr, m_arr)
        rmse = compute_rmse(y_true, preds)
        mae = compute_mae(y_true, preds)

        ranking_metrics = evaluate_ranking_and_diversity(
            df_probe,
            preds,
            item_popularity_counts=pop_counts,
            total_catalog_items=catalog_size,
        )

        results[name] = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            **ranking_metrics,
        }
        logger.info(f"Model: {name:<12} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | NDCG@10: {ranking_metrics.get('NDCG@10', 0):.4f}")

    if ensemble_preds is not None:
        rmse_ens = compute_rmse(y_true, ensemble_preds)
        mae_ens = compute_mae(y_true, ensemble_preds)
        ranking_ens = evaluate_ranking_and_diversity(
            df_probe,
            ensemble_preds,
            item_popularity_counts=pop_counts,
            total_catalog_items=catalog_size,
        )
        results["Ensemble"] = {
            "RMSE": round(rmse_ens, 4),
            "MAE": round(mae_ens, 4),
            **ranking_ens,
        }
        logger.info(f"Model: {'Ensemble':<12} | RMSE: {rmse_ens:.4f} | MAE: {mae_ens:.4f} | NDCG@10: {ranking_ens.get('NDCG@10', 0):.4f}")

    return results
