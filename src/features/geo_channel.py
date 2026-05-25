"""
Groups F (channel & technology) and G (geography).
EDA finding: tokenized_share business HIGHER (58.5% vs 38.5%) — SaaS via Apple Pay.
"""
import polars as pl
from src.config import B2B_MCC_CODES

_B2B = [str(c) for c in B2B_MCC_CODES]


def compute_geo_channel_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("mcc").is_in(_B2B).alias("_is_b2b"),
        (pl.col("country") != "KZ").alias("_is_foreign"),
        (pl.col("channel") == "online").alias("_is_online"),
        (pl.col("channel") == "offline").alias("_is_offline"),
    ])

    agg = df.group_by("card_number").agg([
        pl.len().alias("_n"),
        pl.col("_is_online").sum().alias("_n_online"),
        pl.col("_is_offline").sum().alias("_n_offline"),
        pl.col("tokenized").sum().alias("_n_tok"),
        (pl.col("_is_online") & pl.col("_is_b2b")).sum().alias("_n_online_b2b"),
        pl.col("country").n_unique().alias("n_unique_countries"),
        pl.col("_is_foreign").sum().alias("_n_foreign"),
        (pl.col("_is_foreign") & pl.col("_is_b2b")).sum().alias("_n_foreign_b2b"),
        (pl.col("country") == "KZ").sum().alias("_n_kz"),
    ])

    return agg.with_columns([
        (pl.col("_n_online") / pl.col("_n")).alias("online_share"),
        (pl.col("_n_offline") / pl.col("_n")).alias("offline_share"),
        (pl.col("_n_tok") / pl.col("_n")).alias("tokenized_share"),
        (pl.col("_n_online_b2b") / pl.col("_n")).alias("online_b2b_share"),
        (pl.col("_n_foreign") / pl.col("_n")).alias("foreign_tx_share"),
        (pl.col("_n_kz") / pl.col("_n")).alias("kz_share"),
        pl.when(pl.col("_n_foreign") > 0)
        .then(pl.col("_n_foreign_b2b") / pl.col("_n_foreign"))
        .otherwise(0.0)
        .alias("foreign_b2b_share"),
    ]).select([
        "card_number", "online_share", "offline_share", "tokenized_share",
        "online_b2b_share", "n_unique_countries", "foreign_tx_share",
        "foreign_b2b_share", "kz_share",
    ])
