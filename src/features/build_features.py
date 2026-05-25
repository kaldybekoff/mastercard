"""
build_features.py — orchestrates the full feature matrix construction.
Loads raw parquet → normalizes → calls feature modules → joins → saves.
Run directly: python -m src.features.build_features
"""
from pathlib import Path
import polars as pl

from src.config import (
    BUSINESS_CARDS_PATH, CONSUMER_CARDS_PATH,
    FEATURE_MATRIX_PATH,
)
from src.features.transactional import compute_transactional_features
from src.features.mcc_based import compute_mcc_features
from src.features.temporal import compute_temporal_features
from src.features.recurring import compute_recurring_features
from src.features.geo_channel import compute_geo_channel_features
from src.features.graph_features import compute_graph_features


def _load(path: Path, label: int) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "Is_recurring" in df.columns:
        df = df.rename({"Is_recurring": "is_recurring"})
    if df.schema.get("mcc") != pl.String:
        df = df.with_columns(pl.col("mcc").cast(pl.String))
    return df.with_columns(pl.lit(label).cast(pl.Int8).alias("label"))


def build_feature_matrix(verbose: bool = True) -> pl.DataFrame:
    if verbose:
        print("Loading parquet files...")
    biz = _load(BUSINESS_CARDS_PATH, label=1)
    con = _load(CONSUMER_CARDS_PATH, label=0)

    business_merchants = set(biz["merchant_id"].unique().to_list())
    consumer_merchants = set(con["merchant_id"].unique().to_list())

    df = pl.concat([biz, con])
    if verbose:
        print(f"  Transactions: {len(df):,}  |  Cards: {df['card_number'].n_unique():,}")

    # Card-level labels (all txs for a card share one label)
    card_labels = df.select(["card_number", "label"]).unique(subset=["card_number"])

    if verbose:
        print("Computing feature groups...")

    groups = [
        ("transactional (A/B/H/I)", compute_transactional_features(df)),
        ("mcc-based (C)", compute_mcc_features(df)),
        ("temporal (D)", compute_temporal_features(df)),
        ("recurring (E)", compute_recurring_features(df)),
        ("geo/channel (F/G)", compute_geo_channel_features(df)),
        ("graph (J)", compute_graph_features(df, business_merchants, consumer_merchants)),
    ]

    result = card_labels
    for name, fdf in groups:
        if verbose:
            print(f"  Joining {name}  ({fdf.shape[1] - 1} features)")
        result = result.join(fdf, on="card_number", how="left")

    if verbose:
        print(f"\nFeature matrix: {result.shape[0]:,} cards × {result.shape[1]} columns")
        null_cols = [c for c in result.columns if result[c].null_count() > 0]
        if null_cols:
            print(f"  Columns with nulls: {null_cols}")

    FEATURE_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(FEATURE_MATRIX_PATH)
    if verbose:
        print(f"Saved → {FEATURE_MATRIX_PATH}")

    return result


if __name__ == "__main__":
    build_feature_matrix()
