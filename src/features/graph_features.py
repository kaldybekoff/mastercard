"""
Group J: Merchant Graph Features.
Bipartite card ↔ merchant graph: overlap with business/consumer merchant sets
and cosine similarity to mean business card merchant vector.
"""
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def compute_graph_features(
    df: pl.DataFrame,
    business_merchants: set,
    consumer_merchants: set,
) -> pl.DataFrame:
    # ── Merchant overlap fractions (pure Polars, no UDF) ─────────────────────
    overlaps = (
        df.select(["card_number", "merchant_id"])
        .unique()
        .with_columns([
            pl.col("merchant_id").is_in(list(business_merchants)).alias("_in_biz"),
            pl.col("merchant_id").is_in(list(consumer_merchants)).alias("_in_con"),
        ])
        .group_by("card_number")
        .agg([
            pl.col("_in_biz").sum().alias("_n_biz"),
            pl.col("_in_con").sum().alias("_n_con"),
            pl.len().alias("_n_merch"),
        ])
        .with_columns([
            (pl.col("_n_biz") / pl.col("_n_merch")).alias("business_merchant_overlap"),
            (pl.col("_n_con") / pl.col("_n_merch")).alias("consumer_merchant_overlap"),
        ])
        .select(["card_number", "business_merchant_overlap", "consumer_merchant_overlap"])
    )

    # ── Cosine similarity to mean business card merchant vector ──────────────
    cosine_df = _compute_cosine_similarity(df)

    return overlaps.join(cosine_df, on="card_number", how="left")


def _compute_cosine_similarity(df: pl.DataFrame) -> pl.DataFrame:
    """TF-IDF-style cosine similarity: each card vs mean business card vector."""
    pairs = df.select(["card_number", "merchant_id", "label"]).unique(
        subset=["card_number", "merchant_id"]
    )

    all_cards = pairs["card_number"].unique().to_list()
    all_merchants = pairs["merchant_id"].unique().to_list()
    card_idx = {c: i for i, c in enumerate(all_cards)}
    merch_idx = {m: i for i, m in enumerate(all_merchants)}

    card_col = pairs["card_number"].to_list()
    merch_col = pairs["merchant_id"].to_list()
    rows = np.fromiter((card_idx[c] for c in card_col), dtype=np.int32, count=len(card_col))
    cols = np.fromiter((merch_idx[m] for m in merch_col), dtype=np.int32, count=len(merch_col))

    mat = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(all_cards), len(all_merchants)),
    )

    # Mean business card vector (label == 1)
    biz_labels = (
        df.select(["card_number", "label"])
        .unique(subset=["card_number"])
        .filter(pl.col("label") == 1)
    )
    biz_idxs = [card_idx[c] for c in biz_labels["card_number"].to_list() if c in card_idx]
    mean_biz = np.asarray(mat[biz_idxs].mean(axis=0))  # shape (1, n_merchants)

    sims = cosine_similarity(mat, mean_biz).flatten()

    return pl.DataFrame({
        "card_number": all_cards,
        "merchant_signature_cosine": sims.tolist(),
    })
