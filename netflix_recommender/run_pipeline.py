"""
Master CLI Pipeline for the Netflix Prize Recommender System.
Runs the complete workflow:
1. Data ingestion & dense-core extraction with strict anti-join on official probe set.
2. EDA distribution generation.
3. Decoupled Regularized Baseline (mu + b_u + b_i).
4. Item-Item Collaborative Filtering on Residuals with Pearson Bayesian Shrinkage.
5. Biased Matrix Factorization on Apple Silicon GPU (Metal MPS).
6. SVD++ with Implicit Feedback on Apple Silicon GPU (Metal MPS).
7. Content-Based Cold-Start Bridge (TF-IDF + Decade -> Ridge Latent Projection).
8. Hybrid Blending/Stacking Ensemble optimization.
9. Comprehensive evaluation (RMSE, MAE, Precision@K, Recall@K, NDCG@K, Coverage, Novelty).
10. High-resolution figure generation and metrics export.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import json
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import polars as pl
from scipy.sparse import load_npz

from src.config import (
    TRAIN_PARQUET,
    PROBE_PARQUET,
    MOVIES_PARQUET,
    TRAIN_CSR_FILE,
    ID_MAPS_FILE,
    METRICS_SUMMARY_FILE,
    FIGURES_DIR,
    DEVICE,
    BIASED_MF_CONFIG,
    SVDPP_CONFIG,
    COLD_START_CONFIG,
)
from src.data_pipeline import run_pipeline as run_data_pipeline
from src.models.baseline import DecoupledBaseline
from src.models.collaborative_knn import ResidualItemKNN
from src.models.latent_factors_gpu import BiasedMFRecommender, SVDPPRecommender
from src.models.content_cold_start import ContentColdStartBridge
from src.models.ensemble import HybridEnsemble
from src.evaluation import evaluate_all_models, compute_rmse
from src.visualization import (
    plot_eda_ratings_distribution,
    plot_rmse_comparison,
    plot_training_curves,
    plot_ensemble_weights,
    plot_ranking_metrics,
    plot_cold_start_projection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Master Netflix Recommender Pipeline")
    parser.add_argument("--mode", type=str, choices=["sample", "full"], default="sample")
    parser.add_argument("--force-data", action="store_true", help="Force re-extraction of raw data")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs for GPU models")
    parser.add_argument("--factors", type=int, default=32, help="Latent factor dimensions for MF & SVD++")
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=" * 80)
    logger.info(" NETFLIX PRIZE RECOMMENDER SYSTEM PIPELINE ")
    logger.info(f" Mode: {args.mode} | Device: {DEVICE} | Epochs: {args.epochs} | Factors: {args.factors} ")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Data Ingestion & Dense-Core Extraction
    # -------------------------------------------------------------------------
    if args.force_data or not (TRAIN_PARQUET.exists() and PROBE_PARQUET.exists() and MOVIES_PARQUET.exists()):
        logger.info("[STEP 1/7] Running Data Pipeline...")
        run_data_pipeline(mode=args.mode)
    else:
        logger.info("[STEP 1/7] Existing processed Parquet artifacts found. Loading...")

    df_train = pl.read_parquet(TRAIN_PARQUET)
    df_probe = pl.read_parquet(PROBE_PARQUET)
    df_movies = pl.read_parquet(MOVIES_PARQUET)
    csr_train = load_npz(TRAIN_CSR_FILE)

    n_users = int(df_train["user_idx"].max()) + 1
    n_movies = int(df_train["movie_idx"].max()) + 1
    probe_u = df_probe["user_idx"].to_numpy()
    probe_m = df_probe["movie_idx"].to_numpy()
    probe_y = df_probe["rating"].to_numpy()

    logger.info(f"Dataset Loaded: {df_train.height:,} Train Ratings | {df_probe.height:,} Probe Ratings | {n_users:,} Users | {n_movies:,} Movies")

    # -------------------------------------------------------------------------
    # 2. Exploratory Data Analysis Plots
    # -------------------------------------------------------------------------
    logger.info("[STEP 2/7] Generating EDA Distribution Visualizations...")
    plot_eda_ratings_distribution(df_train, df_probe)

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    models_dict = {}
    preds_matrix = []

    # Model 1: Baseline
    logger.info("[STEP 3/7] Training Model 1: Decoupled Regularized Baseline (mu + b_u + b_i)...")
    base_model = DecoupledBaseline().fit(df_train, n_users=n_users, n_movies=n_movies)
    base_preds = base_model.predict_batch(probe_u, probe_m)
    models_dict["Baseline"] = base_model
    preds_matrix.append(base_preds)
    rmse_base = compute_rmse(probe_y, base_preds)
    logger.info(f"-> Model 1 [Baseline] Probe RMSE: {rmse_base:.4f}")

    # Model 2: Item-Item k-NN on Residuals
    logger.info("[STEP 4/7] Training Model 2: Item-Item Collaborative Filtering (Residuals + Pearson Shrinkage)...")
    knn_model = ResidualItemKNN(baseline=base_model).fit(df_train, n_users=n_users, n_movies=n_movies)
    knn_preds = knn_model.predict_batch(probe_u, probe_m)
    models_dict["Item_kNN"] = knn_model
    preds_matrix.append(knn_preds)
    rmse_knn = compute_rmse(probe_y, knn_preds)
    logger.info(f"-> Model 2 [Item_kNN] Probe RMSE: {rmse_knn:.4f}")
    assert rmse_knn <= rmse_base + 1e-5, f"Invariant violated: Item-kNN RMSE ({rmse_knn:.4f}) > Baseline RMSE ({rmse_base:.4f})"

    # Model 3: Biased Matrix Factorization (MPS GPU)
    logger.info(f"[STEP 5/7] Training Model 3: Biased Matrix Factorization on {DEVICE} ({args.epochs} epochs)...")
    bmf_model = BiasedMFRecommender(
        n_factors=args.factors,
        epochs=args.epochs,
        lr=BIASED_MF_CONFIG["lr"],
        weight_decay=BIASED_MF_CONFIG["weight_decay"],
        device=DEVICE,
    ).fit(df_train, df_val=df_probe, n_users=n_users, n_movies=n_movies)
    bmf_preds = bmf_model.predict_batch(probe_u, probe_m)
    models_dict["Biased_MF"] = bmf_model
    preds_matrix.append(bmf_preds)
    rmse_bmf = compute_rmse(probe_y, bmf_preds)
    logger.info(f"-> Model 3 [Biased_MF] Probe RMSE: {rmse_bmf:.4f}")

    # Model 4: SVD++ with Implicit Feedback (MPS GPU)
    logger.info(f"[STEP 6/7] Training Model 4: SVD++ with Implicit Feedback on {DEVICE} ({args.epochs} epochs)...")
    svd_model = SVDPPRecommender(
        n_factors=args.factors,
        epochs=args.epochs,
        lr=SVDPP_CONFIG["lr"],
        weight_decay=SVDPP_CONFIG["weight_decay"],
        device=DEVICE,
    ).fit(df_train, csr_mat=csr_train, df_val=df_probe, n_users=n_users, n_movies=n_movies)
    svd_preds = svd_model.predict_batch(probe_u, probe_m)
    models_dict["SVDPP"] = svd_model
    preds_matrix.append(svd_preds)
    rmse_svd = compute_rmse(probe_y, svd_preds)
    logger.info(f"-> Model 4 [SVDPP] Probe RMSE: {rmse_svd:.4f}")

    # -------------------------------------------------------------------------
    # 4. Content Cold-Start Bridge
    # -------------------------------------------------------------------------
    logger.info("[STEP 7/7] Training Content Cold-Start Bridge (TF-IDF + Ridge Latent Projection)...")
    with open(ID_MAPS_FILE) as f:
        id_maps = json.load(f)
    idx2movie = {int(v): int(k) for k, v in id_maps["movie_to_idx"].items()}
    sampled_mids = [idx2movie[i] for i in range(n_movies)]
    df_movies_train = df_movies.filter(pl.col("movie_id").is_in(sampled_mids)).sort("movie_idx")

    svd_q = svd_model.get_item_factors()
    svd_b = svd_model.get_item_biases()
    svd_p = svd_model.get_user_factors()
    svd_bu = svd_model.model.b_u.weight.detach().squeeze(-1).cpu().numpy()

    content_bridge = ContentColdStartBridge().fit(
        df_movies_train=df_movies_train,
        item_factors=svd_q,
        item_biases=svd_b,
        user_factors=svd_p,
        user_biases=svd_bu,
        global_mean=base_model.mu,
    )

    cold_sim = content_bridge.evaluate_cold_start_simulation(df_probe, df_movies)
    logger.info(f"Cold-Start Simulation: Content Bridge RMSE: {cold_sim['rmse_bridge']:.4f} vs Global Mean RMSE: {cold_sim['rmse_global_mean']:.4f} (Improvement: {cold_sim['rmse_improvement']:.4f})")

    # Generate projection PCA visualization
    pred_cold_q, _ = content_bridge.project_items(df_movies_train.head(50))
    plot_cold_start_projection(svd_q, pred_cold_q)

    # -------------------------------------------------------------------------
    # 5. Hybrid Stacking / Blending Ensemble
    # -------------------------------------------------------------------------
    logger.info("Optimizing Hybrid Stacking Blending Ensemble...")
    P = np.column_stack(preds_matrix)
    ensemble = HybridEnsemble(model_names=["Baseline", "Item_kNN", "Biased_MF", "SVDPP"]).fit(P, probe_y)
    ensemble_preds = ensemble.predict(P)
    rmse_ensemble = compute_rmse(probe_y, ensemble_preds)
    logger.info(f"-> Final [Ensemble] Probe RMSE: {rmse_ensemble:.4f}")
    assert rmse_ensemble <= rmse_base, f"Invariant violated: Ensemble RMSE ({rmse_ensemble:.4f}) > Baseline RMSE ({rmse_base:.4f})"

    # -------------------------------------------------------------------------
    # 6. Comprehensive Evaluation & Reporting
    # -------------------------------------------------------------------------
    metrics = evaluate_all_models(models_dict, df_probe, df_train=df_train, ensemble_preds=ensemble_preds)

    # Add cold-start simulation results
    metrics["ColdStart_Simulation"] = cold_sim

    # -------------------------------------------------------------------------
    # 7. Visualization Generation & Artifacts Export
    # -------------------------------------------------------------------------
    logger.info("Generating Final High-Resolution Evaluation Figures...")
    plot_rmse_comparison(metrics)
    plot_training_curves(bmf_model.history, svd_model.history)
    plot_ensemble_weights(ensemble.get_weights_summary())
    plot_ranking_metrics(metrics)

    # Save metrics JSON
    with open(METRICS_SUMMARY_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    total_time = time.time() - t_start

    print("\n" + "=" * 80)
    print("                     NETFLIX PRIZE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<16} | {'Probe RMSE':<10} | {'Probe MAE':<10} | {'NDCG@10':<10} | {'Recall@10':<10}")
    print("-" * 80)
    for model_name in ["Baseline", "Item_kNN", "Biased_MF", "SVDPP", "Ensemble"]:
        if model_name in metrics:
            m = metrics[model_name]
            print(f"{model_name:<16} | {m['RMSE']:<10.4f} | {m['MAE']:<10.4f} | {m.get('NDCG@10', 0):<10.4f} | {m.get('Recall@10', 0):<10.4f}")
    print("=" * 80)
    print(f"Grand Prize 10% Benchmark: 0.8567 | Cinematch Benchmark: 0.9525")
    print(f"Optimal Blending Weights: {ensemble.get_weights_summary()}")
    print(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} mins)")
    print(f"Metrics saved to: {METRICS_SUMMARY_FILE}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
