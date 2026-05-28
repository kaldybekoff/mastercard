"""
Phase 0: Top-50 qualitative inspection.

Цель: проверить, что топ-50 consumer-карт по PU-bagging score реально
показывают бизнес-паттерны (4 ключевых сигнала из Q4 организаторов):
  1. merchant_hhi          — концентрация по торговцам
  2. b2b_spend_share        — доля B2B MCC
  3. recurring_amount_share — регулярные крупные списания
  4. foreign_tx_share       — трансграничные платежи

Решающий критерий:
  - Если топ-50 имеют B2B>40%, hhi>0.3, recurring>20%, foreign>0.3 → модель работает
  - Если топ-50 неотличимы от среднего consumer → нужна Фаза 1.5 (anomaly boost)
"""
import sys
from pathlib import Path

# Force UTF-8 for stdout to avoid Windows cp1251 issues
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONSUMER_SCORED = ROOT / "data" / "processed" / "consumer_scored.parquet"
FEATURE_MATRIX = ROOT / "data" / "processed" / "feature_matrix.parquet"
REPORT_PATH = ROOT / "reports" / "diagnostics" / "phase0_top50.md"


# 4 ключевых сигнала из Q4 организаторов + наши топ-фичи
KEY_SIGNALS = [
    "merchant_hhi",
    "b2b_spend_share",
    "recurring_amount_share",
    "foreign_tx_share",
    "business_merchant_overlap",
    "weekday_share",
    "evening_share",
    "n_unique_merchants",
    "consumer_merchant_overlap",
]


def main():
    print("=" * 70)
    print("PHASE 0: TOP-50 QUALITATIVE INSPECTION")
    print("=" * 70)

    # Загрузить scored consumer и feature matrix (для медиан business)
    scored = pl.read_parquet(CONSUMER_SCORED)
    fm = pl.read_parquet(FEATURE_MATRIX)

    business_df = fm.filter(pl.col("label") == 1)
    consumer_df = fm.filter(pl.col("label") == 0)

    print(f"\nLoaded:")
    print(f"  scored consumer: {scored.shape}")
    print(f"  business cards:  {business_df.shape}")
    print(f"  consumer cards:  {consumer_df.shape}")

    # Сортировать по score, взять топ-50
    top50 = scored.sort("business_score", descending=True).head(50)

    print(f"\nTop-50 score range: {top50['business_score'].min():.4f} ... {top50['business_score'].max():.4f}")
    print(f"Top-50 score median:  {top50['business_score'].median():.4f}")

    # ─── Сравнение медиан ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON: Top-50 consumer vs typical business vs typical consumer")
    print("=" * 70)

    rows = []
    for feat in KEY_SIGNALS:
        if feat not in scored.columns:
            print(f"  [WARN] {feat} not in scored data, skipping")
            continue
        top50_med = float(top50[feat].median())
        biz_med = float(business_df[feat].median())
        con_med = float(consumer_df[feat].median())

        # На сколько top-50 ближе к business или к consumer?
        # Если biz_med != con_med: position = (top50 - con) / (biz - con)
        # = 0 → как consumer, 1 → как business, между ними — частичный сдвиг
        if abs(biz_med - con_med) > 1e-6:
            position = (top50_med - con_med) / (biz_med - con_med)
        else:
            position = float("nan")

        rows.append(
            {
                "feature": feat,
                "top50_median": top50_med,
                "business_median": biz_med,
                "consumer_median": con_med,
                "position": position,
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print()
    print(df.to_string(index=False))

    # ─── Вердикт ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # 4 главных сигнала из Q4
    q4_signals = ["merchant_hhi", "b2b_spend_share", "recurring_amount_share", "foreign_tx_share"]
    q4_df = df[df["feature"].isin(q4_signals)].copy()
    avg_position = q4_df["position"].mean()

    print(f"\nAverage 'business-likeness position' on Q4 signals: {avg_position:.3f}")
    print(f"  (0.0 = looks like consumer, 1.0 = looks like business)")

    if avg_position > 0.7:
        verdict = "MODEL WORKS — top-50 strongly resembles business profile"
        next_step = "Skip Phase 1.5. Go directly to Phase 2 (validation packaging)."
    elif avg_position > 0.4:
        verdict = "MODEL PARTIALLY WORKS — top-50 sits between business and consumer"
        next_step = "Consider Phase 1.5 (anomaly boost) to sharpen ranking, then Phase 2."
    else:
        verdict = "MODEL BROKEN — top-50 looks indistinguishable from typical consumer"
        next_step = "MANDATORY: Phase 1.5 (anomaly boost) before submission."

    print(f"\n  VERDICT: {verdict}")
    print(f"  NEXT:    {next_step}")

    # ─── Per-card detail ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP-20 CARDS DETAIL")
    print("=" * 70)
    detail_cols = ["card_number", "business_score"] + KEY_SIGNALS
    detail_cols = [c for c in detail_cols if c in scored.columns]
    top20 = top50.select(detail_cols).head(20).to_pandas()
    print()
    print(top20.to_string(index=False))

    # ─── Сохранить report в Markdown ──────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Phase 0 — Top-50 Inspection Report\n\n")
        f.write(f"**Verdict:** {verdict}\n\n")
        f.write(f"**Average business-likeness position on Q4 signals:** {avg_position:.3f}\n\n")
        f.write(f"**Next step:** {next_step}\n\n")
        f.write("## Feature comparison\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Top-20 detail\n\n")
        f.write(top20.to_markdown(index=False, floatfmt=".4f"))

    print(f"\nReport saved: {REPORT_PATH}")
    return verdict, avg_position


if __name__ == "__main__":
    main()
