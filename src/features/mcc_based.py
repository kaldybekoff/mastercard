"""
Group C: MCC-based business signals.
mcc column is String in the raw data — all comparisons use str codes.
"""
import polars as pl
from src.config import B2B_MCC_CODES, CONSUMER_MCC_CODES, MIXED_MCC_CODES

_B2B = [str(c) for c in B2B_MCC_CODES]
_CONSUMER = [str(c) for c in CONSUMER_MCC_CODES]
_MIXED = [str(c) for c in MIXED_MCC_CODES]


def compute_mcc_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("mcc").is_in(_B2B).alias("_is_b2b"),
        pl.col("mcc").is_in(_CONSUMER).alias("_is_consumer"),
        pl.col("mcc").is_in(_MIXED).alias("_is_mixed"),
        pl.col("transaction_amount_kzt").cast(pl.Float64).alias("_amt"),
        (pl.col("country") != "KZ").alias("_is_foreign"),
    ])

    totals = df.group_by("card_number").agg([
        pl.len().alias("_total_tx"),
        pl.col("_amt").sum().alias("_total_spend"),
    ])

    agg = df.group_by("card_number").agg([
        pl.col("_is_b2b").sum().alias("b2b_tx_count"),
        pl.col("_is_consumer").sum().alias("_consumer_tx"),
        pl.col("_is_mixed").sum().alias("_mixed_tx"),
        pl.when(pl.col("_is_b2b")).then(pl.col("_amt")).otherwise(None).sum().alias("_b2b_spend"),
        pl.when(pl.col("_is_consumer")).then(pl.col("_amt")).otherwise(None).sum().alias("_consumer_spend"),
        pl.when(pl.col("_is_mixed")).then(pl.col("_amt")).otherwise(None).sum().alias("_mixed_spend"),
        (pl.col("_is_b2b") & pl.col("is_recurring")).sum().alias("_b2b_recurring_tx"),
        (pl.col("_is_b2b") & pl.col("_is_foreign")).sum().alias("_b2b_foreign_tx"),
    ])

    # n_unique_b2b_mcc — filter first to avoid counting null as a category
    b2b_mcc_uniq = (
        df.filter(pl.col("_is_b2b"))
        .group_by("card_number")
        .agg(pl.col("mcc").n_unique().alias("n_unique_b2b_mcc"))
    )

    result = (
        agg.join(totals, on="card_number")
        .join(b2b_mcc_uniq, on="card_number", how="left")
        .with_columns([
            pl.col("n_unique_b2b_mcc").fill_null(0),
            (pl.col("b2b_tx_count") / pl.col("_total_tx")).alias("b2b_tx_share"),
            (pl.col("_b2b_spend").fill_null(0) / pl.col("_total_spend")).alias("b2b_spend_share"),
            (pl.col("_consumer_tx") / pl.col("_total_tx")).alias("consumer_tx_share"),
            (pl.col("_consumer_spend").fill_null(0) / pl.col("_total_spend")).alias("consumer_spend_share"),
            (pl.col("_mixed_tx") / pl.col("_total_tx")).alias("mixed_tx_share"),
            (pl.col("_mixed_spend").fill_null(0) / pl.col("_total_spend")).alias("mixed_spend_share"),
            pl.when(pl.col("b2b_tx_count") > 0)
            .then(pl.col("_b2b_recurring_tx") / pl.col("b2b_tx_count"))
            .otherwise(0.0)
            .alias("b2b_recurring_share"),
            pl.when(pl.col("b2b_tx_count") > 0)
            .then(pl.col("_b2b_foreign_tx") / pl.col("b2b_tx_count"))
            .otherwise(0.0)
            .alias("b2b_foreign_share"),
        ])
        .select([
            "card_number", "b2b_tx_count", "b2b_tx_share", "b2b_spend_share",
            "n_unique_b2b_mcc", "consumer_tx_share", "consumer_spend_share",
            "mixed_tx_share", "mixed_spend_share", "b2b_recurring_share", "b2b_foreign_share",
        ])
    )
    return result
