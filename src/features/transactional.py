"""
Groups A (volume/intensity), B (diversity/concentration), H (velocity/burst), I (card-level static).
"""
import polars as pl


def compute_transactional_features(df: pl.DataFrame) -> pl.DataFrame:
    # ── Groups A + B + I: base per-card aggregations ─────────────────────────
    base = df.group_by("card_number").agg([
        pl.col("transaction_amount_kzt").cast(pl.Float64).sum().alias("total_spend_kzt"),
        pl.len().alias("n_transactions"),
        pl.col("transaction_date").n_unique().alias("n_active_days"),
        pl.col("transaction_amount_kzt").cast(pl.Float64).mean().alias("avg_ticket_kzt"),
        pl.col("transaction_amount_kzt").cast(pl.Float64).median().alias("median_ticket_kzt"),
        pl.col("transaction_amount_kzt").cast(pl.Float64).max().alias("max_ticket_kzt"),
        pl.col("transaction_amount_kzt").cast(pl.Float64).std().alias("std_ticket_kzt"),
        pl.col("transaction_amount_kzt").cast(pl.Float64).quantile(0.95).alias("p95_ticket_kzt"),
        pl.col("merchant_id").n_unique().alias("n_unique_merchants"),
        pl.col("mcc").n_unique().alias("n_unique_mcc"),
        pl.col("card_tier").first(),
        pl.col("bank_name").first(),
        pl.col("transaction_date").dt.month().n_unique().alias("n_months_active"),
    ])

    base = base.with_columns([
        (pl.col("n_transactions") / pl.col("n_active_days")).alias("tx_per_active_day"),
        pl.when(pl.col("median_ticket_kzt") > 0)
        .then(pl.col("p95_ticket_kzt") / pl.col("median_ticket_kzt"))
        .otherwise(None)
        .alias("p95_p50_ratio"),
    ])

    # ── Group B: Merchant concentration (HHI, top-K shares) ──────────────────
    merchant_spend = (
        df.group_by(["card_number", "merchant_id"])
        .agg(pl.col("transaction_amount_kzt").cast(pl.Float64).sum().alias("merch_spend"))
    )
    card_total = (
        df.group_by("card_number")
        .agg(pl.col("transaction_amount_kzt").cast(pl.Float64).sum().alias("_card_total"))
    )
    merchant_spend = merchant_spend.join(card_total, on="card_number").with_columns(
        (pl.col("merch_spend") / pl.col("_card_total")).alias("share")
    )

    concentration = merchant_spend.group_by("card_number").agg([
        (pl.col("share") ** 2).sum().alias("merchant_hhi"),
        pl.col("share").sort(descending=True).slice(0, 1).sum().alias("top1_merchant_share"),
        pl.col("share").sort(descending=True).slice(0, 5).sum().alias("top5_merchants_share"),
    ])

    monthly_merchants = (
        df.with_columns(pl.col("transaction_date").dt.month().alias("_month"))
        .group_by(["card_number", "_month"])
        .agg(pl.col("merchant_id").n_unique().alias("_n_merch"))
        .group_by("card_number")
        .agg(pl.col("_n_merch").mean().alias("unique_merchants_per_month"))
    )

    # ── Group H: Velocity / Burst ─────────────────────────────────────────────
    daily = (
        df.group_by(["card_number", "transaction_date"])
        .agg([
            pl.len().alias("_daily_tx"),
            pl.col("transaction_amount_kzt").cast(pl.Float64).sum().alias("_daily_spend"),
        ])
    )
    burst = daily.group_by("card_number").agg([
        pl.col("_daily_tx").max().alias("max_daily_tx_count"),
        pl.col("_daily_spend").max().alias("max_daily_spend_kzt"),
        (pl.col("_daily_tx") >= 5).sum().alias("n_days_5plus_tx"),
        (pl.col("_daily_tx") >= 10).sum().alias("n_days_10plus_tx"),
    ])

    return (
        base
        .join(concentration, on="card_number", how="left")
        .join(monthly_merchants, on="card_number", how="left")
        .join(burst, on="card_number", how="left")
    )
