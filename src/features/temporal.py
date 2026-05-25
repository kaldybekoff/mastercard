"""
Group D: Temporal patterns (working hours, weekday/weekend, night, entropy, lunch dip).
Key finding from EDA: business night_share HIGHER (SaaS UTC billing = 2-5am Almaty).
night_recurring_share median for consumer = 0.0 — strongest separator.
"""
import math
import polars as pl
from src.config import (
    B2B_MCC_CODES,
    BUSINESS_HOURS_START, BUSINESS_HOURS_END,
    EVENING_START, EVENING_END,
    NIGHT_START, NIGHT_END,
    MORNING_START, MORNING_END,
    BUSINESS_DAYS, WEEKEND_DAYS,
)

_B2B = [str(c) for c in B2B_MCC_CODES]
_BIZ_DAYS = list(BUSINESS_DAYS)
_WKND_DAYS = list(WEEKEND_DAYS)


def compute_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    # Add time-derived columns
    df = df.with_columns([
        pl.col("transaction_timestamp").dt.hour().alias("_hour"),
        pl.col("transaction_timestamp").dt.weekday().alias("_dow"),  # 0=Mon, 6=Sun
        pl.col("transaction_date").dt.month().alias("_month"),
        pl.col("mcc").is_in(_B2B).alias("_is_b2b"),
    ])

    # Boolean flags — separate with_columns so _hour/_dow/_is_b2b are available
    df = df.with_columns([
        (
            (pl.col("_hour") >= BUSINESS_HOURS_START) &
            (pl.col("_hour") < BUSINESS_HOURS_END) &
            pl.col("_dow").is_in(_BIZ_DAYS)
        ).alias("_biz_hours"),
        pl.col("_dow").is_in(_BIZ_DAYS).alias("_weekday"),
        pl.col("_dow").is_in(_WKND_DAYS).alias("_weekend"),
        (
            (pl.col("_hour") >= EVENING_START) & (pl.col("_hour") < EVENING_END)
        ).alias("_evening"),
        (
            (pl.col("_hour") >= NIGHT_START) | (pl.col("_hour") < NIGHT_END)
        ).alias("_night"),
        (
            (pl.col("_hour") >= MORNING_START) & (pl.col("_hour") < MORNING_END)
        ).alias("_morning"),
        (pl.col("_month") == 12).alias("_december"),
    ])

    # Interaction flags referencing flags from previous with_columns
    df = df.with_columns([
        (pl.col("_night") & pl.col("is_recurring")).alias("_night_recurring"),
        (pl.col("_biz_hours") & pl.col("_is_b2b")).alias("_biz_hours_b2b"),
    ])

    agg = df.group_by("card_number").agg([
        pl.len().alias("_n"),
        pl.col("_biz_hours").sum().alias("_n_biz"),
        pl.col("_weekday").sum().alias("_n_wd"),
        pl.col("_weekend").sum().alias("_n_we"),
        pl.col("_evening").sum().alias("_n_eve"),
        pl.col("_night").sum().alias("_n_night"),
        pl.col("_morning").sum().alias("_n_morn"),
        pl.col("_december").sum().alias("_n_dec"),
        pl.col("_night_recurring").sum().alias("_n_night_rec"),
        pl.col("_biz_hours_b2b").sum().alias("_n_biz_b2b"),
    ])

    shares = agg.with_columns([
        (pl.col("_n_biz") / pl.col("_n")).alias("business_hours_share"),
        (pl.col("_n_wd") / pl.col("_n")).alias("weekday_share"),
        (pl.col("_n_we") / pl.col("_n")).alias("weekend_share"),
        (pl.col("_n_eve") / pl.col("_n")).alias("evening_share"),
        (pl.col("_n_night") / pl.col("_n")).alias("night_share"),
        (pl.col("_n_morn") / pl.col("_n")).alias("morning_share"),
        (pl.col("_n_dec") / pl.col("_n")).alias("december_share"),
        (pl.col("_n_night_rec") / pl.col("_n")).alias("night_recurring_share"),
        (pl.col("_n_biz_b2b") / pl.col("_n")).alias("business_hours_b2b_share"),
    ]).select([
        "card_number", "business_hours_share", "weekday_share", "weekend_share",
        "evening_share", "night_share", "morning_share", "december_share",
        "night_recurring_share", "business_hours_b2b_share",
    ])

    hour_entropy = _compute_entropy(df, "_hour", "hour_entropy")
    dow_entropy = _compute_entropy(df, "_dow", "dow_entropy")
    lunch_dip = _compute_lunch_dip(df)

    return (
        shares
        .join(hour_entropy, on="card_number", how="left")
        .join(dow_entropy, on="card_number", how="left")
        .join(lunch_dip, on="card_number", how="left")
    )


def _compute_entropy(df: pl.DataFrame, bucket_col: str, out_name: str) -> pl.DataFrame:
    counts = df.group_by(["card_number", bucket_col]).agg(pl.len().alias("cnt"))
    totals = counts.group_by("card_number").agg(pl.col("cnt").sum().alias("total"))
    return (
        counts.join(totals, on="card_number")
        .with_columns((pl.col("cnt") / pl.col("total")).alias("p"))
        .with_columns((-pl.col("p") * (pl.col("p") + 1e-10).log(base=math.e)).alias("h"))
        .group_by("card_number")
        .agg(pl.col("h").sum().alias(out_name))
    )


def _compute_lunch_dip(df: pl.DataFrame) -> pl.DataFrame:
    """share@13h / mean(share@12h, share@14h). Business shows dip (0.727), consumer flat (1.0)."""
    hourly = (
        df.group_by(["card_number", "_hour"])
        .agg(pl.len().alias("cnt"))
    )
    totals = hourly.group_by("card_number").agg(pl.col("cnt").sum().alias("total"))
    hourly = hourly.join(totals, on="card_number").with_columns(
        (pl.col("cnt") / pl.col("total")).alias("share")
    )

    base = df.select("card_number").unique()
    for h, col in [(12, "h12"), (13, "h13"), (14, "h14")]:
        h_df = hourly.filter(pl.col("_hour") == h).select(
            ["card_number", pl.col("share").alias(col)]
        )
        base = base.join(h_df, on="card_number", how="left")

    return (
        base
        .with_columns([
            pl.col("h12").fill_null(0.0),
            pl.col("h13").fill_null(0.0),
            pl.col("h14").fill_null(0.0),
        ])
        .with_columns(
            pl.when(((pl.col("h12") + pl.col("h14")) / 2) > 0)
            .then(pl.col("h13") / ((pl.col("h12") + pl.col("h14")) / 2))
            .otherwise(1.0)
            .alias("lunch_dip_ratio")
        )
        .select(["card_number", "lunch_dip_ratio"])
    )
