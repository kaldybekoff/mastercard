"""
Phase 1.5: Anomaly Boost эксперимент.

Гипотеза:
  Скрытый предприниматель = consumer карта, чьё поведение аномально для consumer.
  PU-score говорит "похоже на бизнес".
  Anomaly score говорит "аномально среди consumer".
  Карта-кандидат должна иметь оба.

Алгоритм:
  1. Обучить IsolationForest на 80K consumer (используя те же фичи что и PU)
  2. Получить anomaly_score для каждой карты (выше = более аномальна)
  3. Скомбинировать: combined_rank = sqrt(rank_pu × rank_anomaly)
  4. Сравнить топ-N по combined с топ-N по PU:
       - сколько новых карт добавилось?
       - они выглядят как бизнес?

Output:
  data/processed/consumer_scored_v2.parquet  — с anomaly_score и combined_rank
  submission_combined.csv  — альтернативный сабмишн
  reports/diagnostics/anomaly_experiment.md  — отчёт со сравнением
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import IsolationForest
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_MATRIX_PATH, PROCESSED_DIR, RANDOM_STATE
from src.models.train import prepare_splits

OUT_DIR = ROOT / "reports" / "diagnostics"


def main():
    print("=" * 70)
    print("PHASE 1.5: ANOMALY BOOST EXPERIMENT")
    print("=" * 70)

    fm = pl.read_parquet(FEATURE_MATRIX_PATH)
    consumer_scored = pl.read_parquet(PROCESSED_DIR / "consumer_scored.parquet")
    _, _, _, _, feature_cols = prepare_splits(fm)

    # ─── Train IsolationForest на 80K consumer ──────────────────────
    consumer_df = fm.filter(pl.col("label") == 0).to_pandas()
    X_consumer = consumer_df[feature_cols]

    print(f"\nTraining IsolationForest on {len(X_consumer):,} consumer cards...")
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_consumer)

    # Negate: lower decision_function → more anomalous → boost
    anomaly_raw = -iso.decision_function(X_consumer)
    print(f"  Anomaly score range: [{anomaly_raw.min():.4f}, {anomaly_raw.max():.4f}]")
    print(f"  Mean: {anomaly_raw.mean():.4f}, std: {anomaly_raw.std():.4f}")

    # ─── Align with consumer_scored ─────────────────────────────────
    # consumer_df comes from fm with label==0 in same order
    # consumer_scored has business_score. We need to align by card_number.
    consumer_df["anomaly_score"] = anomaly_raw

    # Merge with scored df
    scored = consumer_scored.to_pandas().merge(
        consumer_df[["card_number", "anomaly_score"]],
        on="card_number",
        how="left",
    )

    # ─── Correlation check ──────────────────────────────────────────
    pu_score = scored["business_score"].values
    anomaly_score = scored["anomaly_score"].values
    spear, pval = spearmanr(pu_score, anomaly_score)
    print(f"\nSpearman rank correlation (PU score, anomaly score): {spear:.4f}  (p={pval:.2e})")

    if spear < 0:
        print("  ⚠️ Negative correlation — anomaly and PU disagree on what's business-like!")
        print("  Anomaly boost may HURT ROC-AUC. Investigate before using.")
    elif spear < 0.1:
        print("  ~ Weak correlation — anomaly adds new signal independent of PU")
    else:
        print("  ✓ Positive correlation — anomaly aligns with PU direction")

    # ─── Build combined score (rank-based geometric mean) ───────────
    n = len(scored)
    pu_rank = rankdata(pu_score) / n
    ano_rank = rankdata(anomaly_score) / n
    combined_rank = np.sqrt(pu_rank * ano_rank)

    scored["pu_rank"] = pu_rank
    scored["anomaly_rank"] = ano_rank
    scored["combined_score"] = combined_rank

    # ─── Compare top-N (PU only) vs top-N (combined) ────────────────
    print("\n" + "=" * 70)
    print("TOP-N COMPARISON: PU only vs Combined")
    print("=" * 70)

    KEY_FEATURES = [
        "merchant_hhi", "b2b_spend_share", "recurring_amount_share",
        "foreign_tx_share", "business_merchant_overlap", "n_unique_merchants",
        "weekday_share", "evening_share",
    ]

    biz_median = fm.filter(pl.col("label") == 1).to_pandas()[KEY_FEATURES].median()
    con_median = fm.filter(pl.col("label") == 0).to_pandas()[KEY_FEATURES].median()

    for n_top in [50, 100, 500, 1000]:
        top_pu = set(scored.nlargest(n_top, "business_score")["card_number"])
        top_combined = set(scored.nlargest(n_top, "combined_score")["card_number"])
        intersect = top_pu & top_combined
        only_pu = top_pu - top_combined
        only_combined = top_combined - top_pu

        print(f"\nTop-{n_top}:")
        print(f"  In both:       {len(intersect):>5}")
        print(f"  Only in PU:    {len(only_pu):>5}")
        print(f"  Only in COMB:  {len(only_combined):>5} ← NEW candidates surfaced by anomaly boost")

        # Profile the "only in combined" cards
        if len(only_combined) > 0:
            new_cards = scored[scored["card_number"].isin(only_combined)]
            print(f"  Profile of NEW cards (median):")
            for feat in KEY_FEATURES:
                val = new_cards[feat].median()
                bm = biz_median[feat]
                cm = con_median[feat]
                # How business-like?
                if abs(bm - cm) > 1e-6:
                    pos = (val - cm) / (bm - cm)
                else:
                    pos = float("nan")
                marker = "✓" if pos > 0.5 else ("?" if pos > 0.2 else "✗")
                print(f"    {feat:<28} val={val:>7.4f}  (biz={bm:>7.4f}, con={cm:>7.4f})  pos={pos:>5.2f} {marker}")

    # ─── Stat: number of "newly visible" cards across thresholds ────
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(f"\n  PU score:       n>0.01: {(scored['business_score'] > 0.01).sum():>5} "
          f"|  n>0.001: {(scored['business_score'] > 0.001).sum():>5}")
    print(f"  Combined:       n>0.01: {(scored['combined_score'] > 0.01).sum():>5} "
          f"|  n>0.001: {(scored['combined_score'] > 0.001).sum():>5}")
    print(f"\n  Combined quantiles:")
    for q in [0.99, 0.995, 0.999, 0.9999, 1.0]:
        print(f"    q={q}: {scored['combined_score'].quantile(q):.4f}")

    # ─── Save artifacts ─────────────────────────────────────────────
    out_parquet = PROCESSED_DIR / "consumer_scored_v2.parquet"
    pl.from_pandas(scored).write_parquet(out_parquet)
    print(f"\nSaved: {out_parquet}")

    sub_combined = scored[["card_number", "combined_score"]].rename(
        columns={"combined_score": "score"}
    ).sort_values("score", ascending=False)
    sub_combined.to_csv(ROOT / "submission_combined.csv", index=False)
    print(f"Saved: submission_combined.csv ({len(sub_combined):,} rows)")

    print(f"\nOriginal submission (PU only): submission.csv (unchanged)")


if __name__ == "__main__":
    main()
