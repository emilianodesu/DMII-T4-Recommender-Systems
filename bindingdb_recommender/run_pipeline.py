#!/usr/bin/env python3
"""Master CLI Pipeline for BindingDB Chemogenomic Recommender System.
Runs data curation, scaffold-based and target-based partitioning,
fits baseline, collaborative k-NN, content chemogenomics, biased MF, Two-Tower, and Gated Ensemble,
evaluates across Warm, Cold-Ligand, and Cold-Target splits,
and generates diagnostic visualizations.

Usage:
    python run_pipeline.py --sample
    python run_pipeline.py --full
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from src.config import (
    PipelineConfig,
    RAW_DATA_PATH,
    OUTPUTS_DIR,
    FIGURES_DIR,
    METRICS_PATH,
    DEFAULT_DEVICE,
)
from src.data_pipeline import (
    load_and_curate_bindingdb,
    split_random,
    split_scaffold,
    split_target,
    save_splits,
)
from src.featurizer import MoleculeFeaturizer, ProteinFeaturizer
from src.models.baseline import RegularizedBaseline
from src.models.collaborative_knn import CollaborativeResidualKNN
from src.models.content_chemogenomic import ContentChemogenomicKNN
from src.models.latent_factors_gpu import BiasedMatrixFactorization, TwoTowerHybridNeuralMF
from src.models.ensemble import GatedHybridEnsemble
from src.evaluation import evaluate_recsys_model
from src.visualization import (
    plot_pic50_distribution,
    plot_benchmark_comparison,
    plot_parity_scatter,
    plot_target_similarity_matrix,
)


def run_pipeline(
    sample_mode: bool = True,
    sample_targets: int = 35,
    device: str = DEFAULT_DEVICE,
    epochs: int = 25,
) -> Dict[str, Any]:
    start_total_time = time.time()
    print("=" * 78)
    print(" BINDINGDB CHEMOGENOMIC RECOMMENDER SYSTEM (TEMA 4)")
    print(" Real experimental IC50 assays from BindingDB")
    print(f" Mode: {'SAMPLE (Top Targets)' if sample_mode else 'FULL DATASET'}")
    print(f" PyTorch Device: {device} | MPS Available: {torch.backends.mps.is_available()}")
    print("=" * 78)

    cfg = PipelineConfig(
        sample_mode=sample_mode,
        sample_top_targets=sample_targets,
    )
    cfg.biased_mf.device = device
    cfg.biased_mf.epochs = epochs
    cfg.two_tower.device = device
    cfg.two_tower.epochs = epochs

    # 1. Load and Curate Data
    t0 = time.time()
    curated_df = load_and_curate_bindingdb(tsv_path=RAW_DATA_PATH, config=cfg)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    curated_path = cfg.processed_dir / "bindingdb_curated.parquet"
    curated_df.write_parquet(curated_path)
    print(f"[Pipeline] Data curated in {time.time() - t0:.2f}s -> {len(curated_df)} unique pairs.")

    # 2. Generate 3 Splits
    print("\n[Pipeline] Partitioning into 3 Chemogenomic Splits...")
    splits = {
        "random_split": split_random(curated_df, cfg.train_ratio, cfg.val_ratio),
        "scaffold_split": split_scaffold(curated_df, cfg.train_ratio, cfg.val_ratio),
        "target_split": split_target(curated_df, cfg.train_ratio, cfg.val_ratio),
    }
    save_splits(splits, cfg.splits_dir)

    # 3. Shared Featurizers
    mol_feat = MoleculeFeaturizer(
        morgan_radius=cfg.content.morgan_radius,
        morgan_nbits=cfg.content.morgan_nbits,
    )
    prot_feat = ProteinFeaturizer(
        kmer_size=cfg.content.protein_kmer_size,
        max_features=cfg.content.protein_tfidf_max_features,
    )

    metrics_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    stored_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    # 4. Train and Evaluate across all 3 Splits
    for split_name, partitions in splits.items():
        print("\n" + "=" * 65)
        print(f" EVALUATING BENCHMARK: {split_name.upper()}")
        print("=" * 65)

        train_df = partitions["train"]
        test_df = partitions["test"]
        print(f" Train size: {len(train_df)} | Test size: {len(test_df)}")

        metrics_summary[split_name] = {}
        stored_predictions[split_name] = {}

        # A) Biochemical Baseline
        print("\n--> [1/6] Fitting Regularized Baseline...")
        baseline = RegularizedBaseline(cfg.baseline)
        baseline.fit(train_df)
        pred_base = baseline.predict_batch(test_df["target_id"], test_df["ligand_smiles"])
        stored_predictions[split_name]["Baseline"] = pred_base
        m_base = evaluate_recsys_model(test_df, pred_base)
        metrics_summary[split_name]["Baseline"] = m_base
        print(f"    Baseline -> RMSE: {m_base['rmse']:.4f} | MAE: {m_base['mae']:.4f} | R2: {m_base['r2']:.4f} | Rho: {m_base['spearman_rho']:.4f} | NDCG@10: {m_base['ndcg@10']:.4f}")

        # B) Collaborative Residual k-NN
        print("\n--> [2/6] Fitting Collaborative Residual k-NN...")
        collab_knn = CollaborativeResidualKNN(cfg.collab_knn, baseline=baseline)
        collab_knn.fit(train_df)
        pred_cknn = collab_knn.predict_batch(test_df["target_id"], test_df["ligand_smiles"])
        stored_predictions[split_name]["Collaborative k-NN"] = pred_cknn
        m_cknn = evaluate_recsys_model(test_df, pred_cknn)
        metrics_summary[split_name]["Collaborative k-NN"] = m_cknn
        print(f"    Collaborative k-NN -> RMSE: {m_cknn['rmse']:.4f} | MAE: {m_cknn['mae']:.4f} | R2: {m_cknn['r2']:.4f} | Rho: {m_cknn['spearman_rho']:.4f} | NDCG@10: {m_cknn['ndcg@10']:.4f}")

        # C) Content Chemogenomic k-NN
        print("\n--> [3/6] Fitting Content Chemogenomic k-NN...")
        content_knn = ContentChemogenomicKNN(
            cfg.content,
            baseline=baseline,
            mol_featurizer=mol_feat,
            prot_featurizer=prot_feat,
        )
        content_knn.fit(train_df)
        pred_cont = content_knn.predict_batch(
            test_df["target_id"],
            test_df["ligand_smiles"],
            sequences=test_df["sequence"],
        )
        stored_predictions[split_name]["Content Chemogenomic"] = pred_cont
        m_cont = evaluate_recsys_model(test_df, pred_cont)
        metrics_summary[split_name]["Content Chemogenomic"] = m_cont
        print(f"    Content k-NN -> RMSE: {m_cont['rmse']:.4f} | MAE: {m_cont['mae']:.4f} | R2: {m_cont['r2']:.4f} | Rho: {m_cont['spearman_rho']:.4f} | NDCG@10: {m_cont['ndcg@10']:.4f}")

        # D) Biased Matrix Factorization (GPU)
        print("\n--> [4/6] Fitting Biased Matrix Factorization (GPU MPS)...")
        biased_mf = BiasedMatrixFactorization(cfg.biased_mf)
        biased_mf.fit(train_df)
        pred_bmf = biased_mf.predict_batch(test_df["target_id"], test_df["ligand_smiles"])
        stored_predictions[split_name]["Biased MF (GPU)"] = pred_bmf
        m_bmf = evaluate_recsys_model(test_df, pred_bmf)
        metrics_summary[split_name]["Biased MF (GPU)"] = m_bmf
        print(f"    Biased MF -> RMSE: {m_bmf['rmse']:.4f} | MAE: {m_bmf['mae']:.4f} | R2: {m_bmf['r2']:.4f} | Rho: {m_bmf['spearman_rho']:.4f} | NDCG@10: {m_bmf['ndcg@10']:.4f}")

        # E) Two-Tower Hybrid Neural MF (GPU)
        print("\n--> [5/6] Fitting Two-Tower Hybrid Neural MF with Embedding Dropout (GPU MPS)...")
        two_tower = TwoTowerHybridNeuralMF(
            cfg.two_tower,
            mol_featurizer=mol_feat,
            prot_featurizer=prot_feat,
        )
        two_tower.fit(train_df)
        pred_twot = two_tower.predict_batch(
            test_df["target_id"],
            test_df["ligand_smiles"],
            sequences=test_df["sequence"],
        )
        stored_predictions[split_name]["Two-Tower Hybrid"] = pred_twot
        m_twot = evaluate_recsys_model(test_df, pred_twot)
        metrics_summary[split_name]["Two-Tower Hybrid"] = m_twot
        print(f"    Two-Tower Hybrid -> RMSE: {m_twot['rmse']:.4f} | MAE: {m_twot['mae']:.4f} | R2: {m_twot['r2']:.4f} | Rho: {m_twot['spearman_rho']:.4f} | NDCG@10: {m_twot['ndcg@10']:.4f}")

        # F) Gated Hybrid Ensemble
        print("\n--> [6/6] Evaluating Gated Hybrid Ensemble...")
        ensemble = GatedHybridEnsemble(
            baseline=baseline,
            collab_knn=collab_knn,
            content_knn=content_knn,
            biased_mf=biased_mf,
            two_tower=two_tower,
        )
        # Mark known entities
        ensemble.known_targets = set(train_df["target_id"].to_list())
        ensemble.known_ligands = set(train_df["ligand_smiles"].to_list())
        ensemble.is_fitted = True

        pred_ens = ensemble.predict_batch(
            test_df["target_id"],
            test_df["ligand_smiles"],
            sequences=test_df["sequence"],
        )
        stored_predictions[split_name]["Gated Hybrid Ensemble"] = pred_ens
        m_ens = evaluate_recsys_model(test_df, pred_ens)
        metrics_summary[split_name]["Gated Hybrid Ensemble"] = m_ens
        print(f"    Gated Ensemble -> RMSE: {m_ens['rmse']:.4f} | MAE: {m_ens['mae']:.4f} | R2: {m_ens['r2']:.4f} | Rho: {m_ens['spearman_rho']:.4f} | NDCG@10: {m_ens['ndcg@10']:.4f}")

    # 5. Save Metrics Summary
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[Pipeline] Saved complete metrics to {METRICS_PATH}")

    # 6. Generate Publication Figures
    print("\n[Pipeline] Generating Diagnostic Figures in outputs/figures/...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Fig 1: pIC50 distribution
    plot_pic50_distribution(curated_df, FIGURES_DIR / "figure1_pIC50_distribution.png")

    # Fig 2: Benchmark comparison
    plot_benchmark_comparison(metrics_summary, FIGURES_DIR / "figure2_model_benchmark_comparison.png")

    # Fig 3: Parity plot on Random Split (Collaborative KNN vs True) and Scaffold Split (Ensemble vs True)
    y_test_rand = splits["random_split"]["test"]["pIC50"].to_numpy()
    plot_parity_scatter(
        y_test_rand,
        stored_predictions["random_split"]["Gated Hybrid Ensemble"],
        model_name="Gated Hybrid Ensemble",
        split_name="random_split",
        output_path=FIGURES_DIR / "figure3_parity_predicted_vs_true.png",
    )

    # Fig 4: Target similarity heatmap from collaborative k-NN
    if collab_knn.target_sim_matrix is not None:
        target_labels = [collab_knn.idx_to_target[i] for i in range(len(collab_knn.idx_to_target))]
        plot_target_similarity_matrix(
            collab_knn.target_sim_matrix,
            target_labels,
            FIGURES_DIR / "figure4_target_similarity_heatmap.png",
        )

    # Print Final Benchmark Summary Table
    print("\n" + "=" * 90)
    print(" FINAL BENCHMARK SUMMARY (TEMA 4 CHEMOGENOMIC RECOMMENDER)")
    print("=" * 90)
    header = f"{'Split':<16} | {'Model':<24} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} | {'Spearman':<8} | {'NDCG@10':<7}"
    print(header)
    print("-" * len(header))
    for sname, sdata in metrics_summary.items():
        for mname, m in sdata.items():
            print(
                f"{sname:<16} | {mname:<24} | {m['rmse']:<7.3f} | {m['mae']:<7.3f} | {m['r2']:<7.3f} | {m['spearman_rho']:<8.3f} | {m['ndcg@10']:<7.3f}"
            )
        print("-" * len(header))

    elapsed = time.time() - start_total_time
    print(f"\n[Pipeline] Completed successfully in {elapsed:.2f} seconds ({elapsed/60:.1f} min).")
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="BindingDB Chemogenomic Recommender System")
    parser.add_argument("--sample", action="store_true", default=True, help="Run on sampled top targets (fast)")
    parser.add_argument("--full", action="store_true", help="Run on full curated BindingDB dataset")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Computation device (mps, cpu, cuda)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for GPU models")
    parser.add_argument("--sample_targets", type=int, default=35, help="Number of top targets in sample mode")
    args = parser.parse_args()

    sample_mode = not args.full
    run_pipeline(
        sample_mode=sample_mode,
        sample_targets=args.sample_targets,
        device=args.device,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
