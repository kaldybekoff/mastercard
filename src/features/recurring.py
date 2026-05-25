"""
Group E: Regularity — recurring payments, inter-transaction time, monthly spend stability.
"""
import polars as pl


def compute_recurring_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("transaction_amount_kzt").cast(pl.Float64).alias("_amt")
    )

    totals = df.group_by("card_number").agg([
        pl.len().alias("_n"),
        pl.col("_amt").sum().alias("_total_spend"),
    ])

    rec_agg = df.group_by("card_number").agg([
        pl.col("is_recurring").sum().alias("_n_rec"),
        pl.when(pl.col("is_recurring")).then(pl.col("_amt")).otherwise(None).sum().alias("_rec_spend"),
    ])

    # n_unique_recurring_merchants: filter first to avoid null merchant counting
    rec_merchants = (
        df.filter(pl.col("is_recurring"))
        .group_by("card_number")
        .agg(pl.col("merchant_id").n_unique().alias("n_unique_recurring_merchants"))
    )

    base = (
        rec_agg.join(totals, on="card_number")
        .join(rec_merchants, on="card_number", how="left")
        .with_columns([
            pl.col("n_unique_recurring_merchants").fill_null(0),
            (pl.col("_n_rec") / pl.col("_n")).alias("recurring_share"),
            (pl.col("_rec_spend").fill_null(0) / pl.col("_total_spend")).alias("recurring_amount_share"),
        ])
        .select([
            "card_number", "recurring_share", "recurring_amount_share", "n_unique_recurring_merchants",
        ])
    )

    # ── Inter-transaction time (median gap and CV) ────────────────────────────
    inter_tx = (
        df.sort(["card_number", "transaction_timestamp"])
        .group_by("card_number", maintain_order=True)
        .agg(
            pl.col("transaction_timestamp").diff().dt.total_seconds().alias("_gaps")
        )
        .explode("_gaps")
        .drop_nulls()
        .group_by("card_number")
        .agg([
            (pl.col("_gaps").median() / 86400).alias("inter_tx_time_median_days"),
            pl.col("_gaps").std().alias("_gap_std"),
            pl.col("_gaps").mean().alias("_gap_mean"),
        ])
        .with_columns(
            pl.when(pl.col("_gap_mean") > 0)
            .then(pl.col("_gap_std") / pl.col("_gap_mean"))
            .otherwise(None)
            .alias("inter_tx_time_cv")
        )
        .select(["card_number", "inter_tx_time_median_days", "inter_tx_time_cv"])
    )

    # ── Monthly spend CV (stability) ─────────────────────────────────────────
    monthly_cv = (
        df.with_columns(pl.col("transaction_date").dt.month().alias("_month"))
        .group_by(["card_number", "_month"])
        .agg(pl.col("_amt").sum().alias("_ms"))
        .group_by("card_number")
        .agg([
            pl.col("_ms").std().alias("_ms_std"),
            pl.col("_ms").mean().alias("_ms_mean"),
        ])
        .with_columns(
            pl.when(pl.col("_ms_mean") > 0)
            .then(pl.col("_ms_std") / pl.col("_ms_mean"))
            .otherwise(None)
            .alias("monthly_spend_cv")
        )
        .select(["card_number", "monthly_spend_cv"])
    )

    return base.join(inter_tx, on="card_number", how="left").join(monthly_cv, on="card_number", how="left")
