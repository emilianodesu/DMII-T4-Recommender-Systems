"""Evaluation module for Chemogenomic Recommender Systems.
Computes:
1) Global Regression Metrics:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R-squared (R2 score)
   - Spearman's Rank Correlation (rho)
2) Target-Level Recommender Ranking Metrics (for active hits with pIC50 >= 6.0):
   - NDCG@K (Normalized Discounted Cumulative Gain at K)
   - Recall@K (Fraction of active hits retrieved in top-K recommendations)
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
from scipy import stats

from src.config import ACTIVE_PIC50_THRESHOLD


def compute_regression_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """Computes RMSE, MAE, R2, and Spearman rank correlation rho."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) == 0:
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "spearman_rho": 0.0}

    # Handle NaNs or Infs gracefully
    valid_mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.all(valid_mask):
        y_t = y_t[valid_mask]
        y_p = y_p[valid_mask]
        if len(y_t) == 0:
            return {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "spearman_rho": 0.0}

    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mae = float(np.mean(np.abs(y_t - y_p)))

    # R2 score
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    ss_res = np.sum((y_t - y_p) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

    # Spearman rank correlation
    if len(y_t) > 1 and np.std(y_p) > 1e-9:
        res = stats.spearmanr(y_t, y_p)
        spearman_rho = float(res.statistic if hasattr(res, "statistic") else res[0])
    else:
        spearman_rho = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman_rho": spearman_rho,
    }


def compute_ranking_metrics_for_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    threshold: float = ACTIVE_PIC50_THRESHOLD,
) -> Tuple[Optional[float], Optional[float]]:
    """Computes NDCG@K and Recall@K for a single target.
    Returns (ndcg_at_k, recall_at_k) or (None, None) if target has no actives or fewer than 2 items.
    """
    if len(y_true) < 2:
        return None, None

    # Binary relevance: active hit if pIC50 >= threshold
    relevance = (y_true >= threshold).astype(np.float64)
    total_actives = np.sum(relevance)
    if total_actives == 0:
        return None, None

    # Rank indices according to predicted affinity descending
    ranked_indices = np.argsort(-y_pred)
    ranked_rel = relevance[ranked_indices]

    # Evaluate top-k
    actual_k = min(k, len(ranked_rel))
    top_k_rel = ranked_rel[:actual_k]

    # Recall@K
    hits = np.sum(top_k_rel)
    recall_at_k = float(hits / total_actives)

    # DCG@K
    discounts = np.log2(np.arange(2, actual_k + 2))
    dcg = np.sum((2.0 ** top_k_rel - 1.0) / discounts)

    # Ideal DCG@K
    ideal_rel = np.sort(relevance)[::-1][:actual_k]
    idcg = np.sum((2.0 ** ideal_rel - 1.0) / discounts)

    ndcg_at_k = float(dcg / idcg) if idcg > 0 else 0.0

    return ndcg_at_k, recall_at_k


def evaluate_recsys_model(
    test_df: pl.DataFrame,
    y_pred: np.ndarray,
    k: int = 10,
    threshold: float = ACTIVE_PIC50_THRESHOLD,
) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics on a test set:
    Regression metrics (RMSE, MAE, R2, Spearman rho) and
    Recommender Ranking metrics (mean NDCG@K, mean Recall@K).
    """
    y_true = test_df["pIC50"].to_numpy().astype(np.float64)
    reg_metrics = compute_regression_metrics(y_true, y_pred)

    targets = test_df["target_id"].to_list()
    target_groups: Dict[str, Tuple[List[float], List[float]]] = {}
    for t, yt, yp in zip(targets, y_true, y_pred):
        if t not in target_groups:
            target_groups[t] = ([], [])
        target_groups[t][0].append(yt)
        target_groups[t][1].append(yp)

    ndcg_list: List[float] = []
    recall_list: List[float] = []

    for t_id, (yt_list, yp_list) in target_groups.items():
        yt_arr = np.array(yt_list, dtype=np.float64)
        yp_arr = np.array(yp_list, dtype=np.float64)
        ndcg_k, rec_k = compute_ranking_metrics_for_target(
            yt_arr, yp_arr, k=k, threshold=threshold
        )
        if ndcg_k is not None and rec_k is not None:
            ndcg_list.append(ndcg_k)
            recall_list.append(rec_k)

    mean_ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    mean_recall = float(np.mean(recall_list)) if recall_list else 0.0

    return {
        "rmse": reg_metrics["rmse"],
        "mae": reg_metrics["mae"],
        "r2": reg_metrics["r2"],
        "spearman_rho": reg_metrics["spearman_rho"],
        f"ndcg@{k}": mean_ndcg,
        f"recall@{k}": mean_recall,
        "n_evaluated_targets": len(ndcg_list),
        "n_samples": len(y_true),
    }
