"""
Phase 0b: Глубокая диагностика подозрительных фичей и middle-band ранжирования.

Что проверяем:
1. foreign_tx_share — почему медиана 1.0 у всех? Реальное распределение
2. consumer_merchant_overlap — то же самое
3. Score distribution на middle-band (карты с рангом 100-5000)
4. Что отличает карту с score=0.001 от карты с score=0.0001?
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONSUMER_SCORED = ROOT / "data" / "processed" / "consumer_scored.parquet"
FEATURE_MATRIX = ROOT / "data" / "processed" / "feature_matrix.parquet"


def print_dist(name, series_pos, series_neg):
    """Print full distribution stats for both classes."""
    print(f"\n{'─' * 70}")
    print(f"Feature: {name}")
    print(f"{'─' * 70}")
    print(f"{'percentile':>10} | {'business':>12} | {'consumer':>12} | {'top-50':>12}")
    for q in [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]:
        b = series_pos.quantile(q)
        c = series_neg.quantile(q)
        print(f"  q={q:.2f}     | {b:>12.4f} | {c:>12.4f} |")


def main():
    print("=" * 70)
    print("PHASE 0b: DEEP DIAGNOSTICS")
    print("=" * 70)

    fm = pl.read_parquet(FEATURE_MATRIX)
    scored = pl.read_parquet(CONSUMER_SCORED)

    biz = fm.filter(pl.col("label") == 1)
    con = fm.filter(pl.col("label") == 0)
    top50 = scored.sort("business_score", descending=True).head(50)

    # ─── Подозрительные фичи ───────────────────────────────────────
    suspicious = ["foreign_tx_share", "consumer_merchant_overlap", "kz_share"]
    for feat in suspicious:
        print(f"\n{'═' * 70}")
        print(f"FEATURE: {feat}")
        print(f"{'═' * 70}")

        biz_vals = biz[feat]
        con_vals = con[feat]
        t50_vals = top50[feat]

        print(f"\n  Business ({len(biz_vals):,}):")
        print(f"    min={biz_vals.min():.4f}  q05={biz_vals.quantile(0.05):.4f}  q25={biz_vals.quantile(0.25):.4f}")
        print(f"    q50={biz_vals.quantile(0.50):.4f}  q75={biz_vals.quantile(0.75):.4f}  q95={biz_vals.quantile(0.95):.4f}  max={biz_vals.max():.4f}")
        print(f"    mean={biz_vals.mean():.4f}  std={biz_vals.std():.4f}")
        print(f"    n_unique values: {biz_vals.n_unique()}")
        print(f"    == 1.0: {(biz_vals == 1.0).sum():,} ({(biz_vals == 1.0).sum() / len(biz_vals) * 100:.1f}%)")
        print(f"    == 0.0: {(biz_vals == 0.0).sum():,} ({(biz_vals == 0.0).sum() / len(biz_vals) * 100:.1f}%)")

        print(f"\n  Consumer ({len(con_vals):,}):")
        print(f"    min={con_vals.min():.4f}  q05={con_vals.quantile(0.05):.4f}  q25={con_vals.quantile(0.25):.4f}")
        print(f"    q50={con_vals.quantile(0.50):.4f}  q75={con_vals.quantile(0.75):.4f}  q95={con_vals.quantile(0.95):.4f}  max={con_vals.max():.4f}")
        print(f"    mean={con_vals.mean():.4f}  std={con_vals.std():.4f}")
        print(f"    n_unique values: {con_vals.n_unique()}")
        print(f"    == 1.0: {(con_vals == 1.0).sum():,} ({(con_vals == 1.0).sum() / len(con_vals) * 100:.1f}%)")
        print(f"    == 0.0: {(con_vals == 0.0).sum():,} ({(con_vals == 0.0).sum() / len(con_vals) * 100:.1f}%)")

        print(f"\n  Top-50 by score:")
        print(f"    min={t50_vals.min():.4f}  max={t50_vals.max():.4f}  median={t50_vals.median():.4f}")

    # ─── Score distribution: middle band ──────────────────────────────
    print(f"\n{'═' * 70}")
    print("SCORE DISTRIBUTION: How does the middle band look?")
    print(f"{'═' * 70}")

    s = scored.sort("business_score", descending=True)
    ranks_of_interest = [10, 50, 100, 500, 1000, 4000, 10000, 40000, 80000]
    print(f"\n  rank | score              | b2b_share | hhi    | recurring | n_merch")
    for r in ranks_of_interest:
        if r > len(s):
            continue
        row = s.row(r - 1, named=True)
        print(f"  {r:>5} | score={row['business_score']:>12.6f} | "
              f"{row['b2b_spend_share']:>9.4f} | "
              f"{row['merchant_hhi']:>6.4f} | "
              f"{row['recurring_amount_share']:>9.4f} | "
              f"{row['n_unique_merchants']:>7}")

    # ─── Score histogram ────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("SCORE HISTOGRAM (linear bins)")
    print(f"{'═' * 70}")
    scores = s["business_score"].to_numpy()
    bins = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"\n  bin                    | n_cards    | %")
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = ((scores >= lo) & (scores < hi)).sum()
        pct = n / len(scores) * 100
        bar = "█" * int(pct / 2)
        print(f"  [{lo:>9.6f}, {hi:>9.6f}) | {n:>10,} | {pct:>5.2f}%  {bar}")

    n_max = (scores >= bins[-1]).sum()
    print(f"  [{bins[-1]:>9.6f}, +inf)      | {n_max:>10,}")


if __name__ == "__main__":
    main()
