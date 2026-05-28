"""
Phase 6d: Ablation test — устойчивость модели.

Удаляем разные подмножества фичей и измеряем падение ROC-AUC через 5-fold CV.
Если AUC устойчиво остаётся ≥0.99 — модель не зависит от одной фичи / одной группы.

Группы для ablation:
  - none (full model) — baseline
  - drop_night — без night_share, night_recurring_share (timezone-suspicious)
  - drop_graph — без 3 граф-фичей (Group J)
  - drop_geo — без foreign_tx_share, kz_share, b2b_foreign_share (Group G)
  - drop_temporal — без всей Group D
  - only_mcc — ТОЛЬКО Group C (B2B MCC фичи)
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_MATRIX_PATH
from src.models.train import prepare_splits
from src.evaluation.cv import run_5fold_cv

OUT_DIR = ROOT / "reports" / "diagnostics"

# Группы фичей по нашей feature engineering структуре
NIGHT_FEATURES = ["night_share", "night_recurring_share"]
GRAPH_FEATURES = ["business_merchant_overlap", "consumer_merchant_overlap", "merchant_signature_cosine"]
GEO_FEATURES = ["foreign_tx_share", "kz_share", "b2b_foreign_share", "n_unique_countries"]
TEMPORAL_FEATURES = [
    "business_hours_share", "weekday_share", "weekend_share", "evening_share",
    "night_share", "morning_share", "hour_entropy", "dow_entropy",
    "business_hours_b2b_share", "night_recurring_share", "lunch_dip_ratio", "december_share",
]
MCC_FEATURES = [
    "b2b_tx_count", "b2b_tx_share", "b2b_spend_share", "n_unique_b2b_mcc",
    "consumer_tx_share", "consumer_spend_share", "mixed_tx_share", "mixed_spend_share",
    "rental_tx_share", "rental_spend_share", "n_unique_rental_mcc",
    "b2b_recurring_share", "b2b_foreign_share",
]


def main():
    print("=" * 70)
    print("PHASE 6d: ABLATION TEST")
    print("=" * 70)

    fm = pl.read_parquet(FEATURE_MATRIX_PATH)
    X_pos_train, _, X_unlabeled, _, feature_cols = prepare_splits(fm)
    print(f"\nFull feature set: {len(feature_cols)} features")

    experiments = [
        ("Full model (baseline)", feature_cols),
        ("Drop night-features", [c for c in feature_cols if c not in NIGHT_FEATURES]),
        ("Drop graph-features (Group J)", [c for c in feature_cols if c not in GRAPH_FEATURES]),
        ("Drop geo-features (Group G)", [c for c in feature_cols if c not in GEO_FEATURES]),
        ("Drop temporal (Group D)", [c for c in feature_cols if c not in TEMPORAL_FEATURES]),
        ("Only MCC-features (Group C)", [c for c in feature_cols if c in MCC_FEATURES]),
        ("Drop top-3 SHAP features", [
            c for c in feature_cols
            if c not in ["business_merchant_overlap", "tokenized_share", "evening_share"]
        ]),
    ]

    results = []
    for name, feats in experiments:
        if len(feats) == 0:
            print(f"\n[{name}] empty feature set — skipping")
            continue
        print(f"\n[{name}] using {len(feats)} features ...")
        Xp = X_pos_train[feats]
        Xu = X_unlabeled[feats]
        cv = run_5fold_cv(Xp, Xu, n_folds=5)
        results.append({
            "experiment": name,
            "n_features": len(feats),
            "roc_auc_mean": cv["roc_auc"].mean(),
            "roc_auc_std": cv["roc_auc"].std(),
            "pr_auc_mean": cv["pr_auc"].mean(),
            "f1_mean": cv["f1"].mean(),
            "precision_mean": cv["precision"].mean(),
            "recall_mean": cv["recall"].mean(),
        })
        print(f"  ROC-AUC: {results[-1]['roc_auc_mean']:.5f} ± {results[-1]['roc_auc_std']:.5f}")

    df = pd.DataFrame(results)
    df["roc_auc_drop"] = df["roc_auc_mean"].iloc[0] - df["roc_auc_mean"]

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
