"""Data Pipeline module for BindingDB Chemogenomic Recommender.
Curates real experimental IC50 values into pIC50, deduplicates bioassays via median,
extracts Bemis-Murcko scaffolds with RDKit, and generates the 3 evaluation splits:
1) random_split (Warm evaluation)
2) scaffold_split (Cold-Ligand evaluation)
3) target_split (Cold-Target evaluation)
"""
from pathlib import Path
from typing import Dict, Tuple, Optional
import polars as pl
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.config import (
    PipelineConfig,
    IC50_CLIP_MIN_NM,
    IC50_CLIP_MAX_NM,
    RANDOM_SEED,
)


def extract_canonical_and_scaffold(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """Validates SMILES, returns (canonical_smiles, murcko_scaffold).
    Returns (None, None) if SMILES cannot be parsed by RDKit.
    """
    if not smiles or not isinstance(smiles, str):
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, None
        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not scaffold:
            scaffold = f"acyclic_{canonical}"
        return canonical, scaffold
    except Exception:
        return None, None


def load_and_curate_bindingdb(
    tsv_path: Path,
    config: Optional[PipelineConfig] = None,
) -> pl.DataFrame:
    """Loads raw BindingDB TSV using Polars, filters invalid records,
    computes pIC50 with clipping, validates chemistry with RDKit, and aggregates
    duplicate assays of the same (Target, Ligand) pair using the median.
    """
    if config is None:
        config = PipelineConfig()

    print(f"[Pipeline] Reading raw TSV from {tsv_path}...")
    # Read raw TSV cleanly without quote issues
    raw_df = pl.read_csv(
        tsv_path,
        separator="\t",
        quote_char=None,
        truncate_ragged_lines=True,
        infer_schema_length=0,
    )

    # Required columns
    col_ic50 = "IC50 (nM)"
    col_smiles = "Ligand SMILES"
    col_target_name = "Target Name"
    col_uniprot = "UniProt (SwissProt) Primary ID of Target Chain 1"
    col_seq = "BindingDB Target Chain Sequence 1"

    needed_cols = [col_ic50, col_smiles, col_target_name, col_uniprot, col_seq]
    for c in needed_cols:
        if c not in raw_df.columns:
            raise ValueError(f"Missing required column in TSV: {c}")

    # Initial non-null filtering
    df = raw_df.select(needed_cols).filter(
        pl.col(col_ic50).is_not_null() & (pl.col(col_ic50).str.strip_chars() != "") &
        pl.col(col_smiles).is_not_null() & (pl.col(col_smiles).str.strip_chars() != "") &
        (
            (pl.col(col_uniprot).is_not_null() & (pl.col(col_uniprot).str.strip_chars() != "")) |
            (pl.col(col_target_name).is_not_null() & (pl.col(col_target_name).str.strip_chars() != ""))
        ) &
        pl.col(col_seq).is_not_null() & (pl.col(col_seq).str.strip_chars() != "")
    )
    print(f"[Pipeline] Raw rows with non-null bioassay fields: {len(df)}")

    # Sanitize IC50 and cast to Float64
    df = df.with_columns(
        pl.col(col_ic50)
        .str.replace_all(r"[><~ =]", "")
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias("ic50_val")
    ).filter(pl.col("ic50_val").is_not_null() & (pl.col("ic50_val") > 0))

    # Compute pIC50 = 9.0 - log10(clip(IC50_nM, 1e-3, 1e7))
    df = df.with_columns(
        (9.0 - (pl.col("ic50_val").clip(IC50_CLIP_MIN_NM, IC50_CLIP_MAX_NM).log10())).alias("pIC50")
    )

    # Resolve target_id (prefer SwissProt UniProt ID, fallback to sanitized Target Name)
    df = df.with_columns(
        pl.when(
            pl.col(col_uniprot).is_not_null() & (pl.col(col_uniprot).str.strip_chars() != "")
        )
        .then(pl.col(col_uniprot).str.strip_chars())
        .otherwise(pl.col(col_target_name).str.strip_chars())
        .alias("target_id"),
        pl.col(col_target_name).str.strip_chars().alias("target_name"),
        pl.col(col_smiles).str.strip_chars().alias("raw_smiles"),
        pl.col(col_seq).str.strip_chars().alias("sequence"),
    )

    # If sample_mode is active, keep top targets with highest assay counts
    if config.sample_mode:
        print(f"[Pipeline] Sample mode enabled: selecting top {config.sample_top_targets} targets.")
        top_targets = (
            df.group_by("target_id")
            .len()
            .sort("len", descending=True)
            .head(config.sample_top_targets)["target_id"]
            .to_list()
        )
        df = df.filter(pl.col("target_id").is_in(top_targets))
        print(f"[Pipeline] Rows after top targets filter: {len(df)}")
    elif config.min_ligands_per_target > 1:
        target_counts = df.group_by("target_id").len()
        valid_targets = target_counts.filter(pl.col("len") >= config.min_ligands_per_target)["target_id"].to_list()
        df = df.filter(pl.col("target_id").is_in(valid_targets))

    # Resolve canonical sequence per target (longest observed sequence)
    target_canonical_seq = (
        df.group_by("target_id")
        .agg(
            pl.col("sequence")
            .sort_by(pl.col("sequence").str.len_bytes(), descending=True)
            .first()
            .alias("canonical_sequence"),
            pl.col("target_name").first().alias("canonical_target_name"),
        )
    )

    # RDKit validation, canonicalization and Bemis-Murcko scaffold extraction on unique SMILES
    unique_smiles = df["raw_smiles"].unique().to_list()
    print(f"[Pipeline] Validating {len(unique_smiles)} unique SMILES via RDKit...")

    smiles_mapping = {}
    for s in unique_smiles:
        can_s, scaf = extract_canonical_and_scaffold(s)
        if can_s is not None:
            smiles_mapping[s] = (can_s, scaf)

    valid_raw_smiles = set(smiles_mapping.keys())
    df = df.filter(pl.col("raw_smiles").is_in(valid_raw_smiles))

    # Map canonical SMILES and scaffold
    can_smiles_list = [smiles_mapping[s][0] for s in df["raw_smiles"]]
    scaffold_list = [smiles_mapping[s][1] for s in df["raw_smiles"]]

    df = df.with_columns(
        pl.Series("ligand_smiles", can_smiles_list),
        pl.Series("scaffold", scaffold_list),
    )

    # Join canonical target sequence and target name
    df = df.drop(["sequence", "target_name"]).join(
        target_canonical_seq, on="target_id", how="left"
    ).rename({
        "canonical_sequence": "sequence",
        "canonical_target_name": "target_name",
    })

    # Bioassay Deduplication: aggregate multiple measurements of the same (Target, Ligand) pair
    # by computing the MEDIAN pIC50
    print(f"[Pipeline] Aggregating bioassay duplicates via pIC50 median...")
    aggregated_df = (
        df.group_by(["target_id", "ligand_smiles"])
        .agg([
            pl.col("pIC50").median().alias("pIC50"),
            pl.col("target_name").first().alias("target_name"),
            pl.col("sequence").first().alias("sequence"),
            pl.col("scaffold").first().alias("scaffold"),
            pl.len().alias("assay_count"),
        ])
        .sort(["target_id", "ligand_smiles"])
    )

    print(
        f"[Pipeline] Curation complete: {len(aggregated_df)} unique pairs "
        f"across {aggregated_df['target_id'].n_unique()} targets "
        f"and {aggregated_df['ligand_smiles'].n_unique()} ligands."
    )
    return aggregated_df


def split_random(
    df: pl.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> Dict[str, pl.DataFrame]:
    """Generates Random Split (Warm benchmark):
    Pairs are randomly assigned to train/val/test.
    """
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = shuffled.slice(0, n_train)
    val_df = shuffled.slice(n_train, n_val)
    test_df = shuffled.slice(n_train + n_val)

    return {"train": train_df, "val": val_df, "test": test_df}


def split_scaffold(
    df: pl.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> Dict[str, pl.DataFrame]:
    """Generates Bemis-Murcko Scaffold Split (Cold-Ligand benchmark):
    Molecules are grouped by Murcko scaffold. Scaffolds are sorted by size
    and partitioned greedily so that train gets ~80%, val ~10%, and test ~10%.
    NO ligand in test shares a scaffold with any ligand in train.
    """
    scaffold_counts = (
        df.group_by("scaffold")
        .len()
        .sort("len", descending=True)
    )
    total_pairs = len(df)
    train_limit = train_ratio * total_pairs
    val_limit = val_ratio * total_pairs

    train_scaffolds = set()
    val_scaffolds = set()
    test_scaffolds = set()

    train_cnt, val_cnt = 0, 0

    for row in scaffold_counts.iter_rows(named=True):
        scaf = row["scaffold"]
        cnt = row["len"]

        if train_cnt + cnt <= train_limit:
            train_scaffolds.add(scaf)
            train_cnt += cnt
        elif val_cnt + cnt <= val_limit:
            val_scaffolds.add(scaf)
            val_cnt += cnt
        else:
            test_scaffolds.add(scaf)

    train_df = df.filter(pl.col("scaffold").is_in(train_scaffolds))
    val_df = df.filter(pl.col("scaffold").is_in(val_scaffolds))
    test_df = df.filter(pl.col("scaffold").is_in(test_scaffolds))

    return {"train": train_df, "val": val_df, "test": test_df}


def split_target(
    df: pl.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> Dict[str, pl.DataFrame]:
    """Generates Target Split (Cold-Target benchmark):
    Proteins are partitioned so that test targets were NEVER seen during training.
    Evaluates zero-shot chemogenomic generalization to new biological targets.
    """
    # Deterministic pseudo-random shuffle of targets with seed
    target_counts = (
        df.group_by("target_id")
        .len()
        .sample(fraction=1.0, shuffle=True, seed=seed)
    )
    total_pairs = len(df)
    train_limit = train_ratio * total_pairs
    val_limit = val_ratio * total_pairs

    train_targets = set()
    val_targets = set()
    test_targets = set()

    train_cnt, val_cnt = 0, 0

    for row in target_counts.iter_rows(named=True):
        t_id = row["target_id"]
        cnt = row["len"]

        if train_cnt + cnt <= train_limit or len(train_targets) < 2:
            train_targets.add(t_id)
            train_cnt += cnt
        elif val_cnt + cnt <= val_limit or len(val_targets) < 1:
            val_targets.add(t_id)
            val_cnt += cnt
        else:
            test_targets.add(t_id)

    # In case test_targets ended up empty due to greedy binning on small target count, rebalance
    if len(test_targets) == 0:
        if len(val_targets) > 1:
            test_targets.add(val_targets.pop())
        elif len(train_targets) > 2:
            test_targets.add(train_targets.pop())

    train_df = df.filter(pl.col("target_id").is_in(train_targets))
    val_df = df.filter(pl.col("target_id").is_in(val_targets))
    test_df = df.filter(pl.col("target_id").is_in(test_targets))

    return {"train": train_df, "val": val_df, "test": test_df}


def save_splits(
    splits_dict: Dict[str, Dict[str, pl.DataFrame]],
    base_dir: Path,
) -> None:
    """Saves generated splits to parquet and csv in base_dir/<split_name>/."""
    base_dir.mkdir(parents=True, exist_ok=True)
    for split_name, partitions in splits_dict.items():
        split_dir = base_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for part_name, part_df in partitions.items():
            parquet_path = split_dir / f"{part_name}.parquet"
            csv_path = split_dir / f"{part_name}.csv"
            part_df.write_parquet(parquet_path)
            part_df.write_csv(csv_path)
            print(f"[Pipeline] Saved {split_name}/{part_name}: {len(part_df)} pairs -> {parquet_path}")


def prepare_and_save_all_splits(
    tsv_path: Path,
    config: PipelineConfig,
) -> Dict[str, Dict[str, pl.DataFrame]]:
    """Master routine: loads raw TSV, curates dataset, computes all 3 splits,
    saves them to disk, and returns the partitions.
    """
    df = load_and_curate_bindingdb(tsv_path=tsv_path, config=config)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    curated_path = config.processed_dir / "bindingdb_curated.parquet"
    df.write_parquet(curated_path)
    print(f"[Pipeline] Saved curated dataset to {curated_path}")

    print("\n[Pipeline] Generating Random Split (Warm)...")
    random_splits = split_random(df, config.train_ratio, config.val_ratio, RANDOM_SEED)

    print("[Pipeline] Generating Bemis-Murcko Scaffold Split (Cold-Ligand)...")
    scaffold_splits = split_scaffold(df, config.train_ratio, config.val_ratio, RANDOM_SEED)

    print("[Pipeline] Generating Target Split (Cold-Target)...")
    target_splits = split_target(df, config.train_ratio, config.val_ratio, RANDOM_SEED)

    all_splits = {
        "random_split": random_splits,
        "scaffold_split": scaffold_splits,
        "target_split": target_splits,
    }

    save_splits(all_splits, config.splits_dir)
    return all_splits
