"""
Retrain PU-Bagging on FIXED feature matrix (post foreign_tx_share bug fix).
Saves model, holdout metrics, scored consumer parquet.
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_MATRIX_PATH, MODELS_DIR, RANDOM_STATE, PROCESSED_DIR
from src.models.pu_bagging import PUBaggingClassifier
from src.models.train import prepare_splits, save_model
from src.models.baseline import build_baseline


def main():
    print("=" * 70)
    print("RETRAIN AFTER FIX")
    print("=" * 70)

    fm = pl.read_parquet(FEATURE_MATRIX_PATH)
    print(f"\nFeature matrix: {fm.shape}")

    X_pos_train, X_pos_holdout, X_unlabeled, X_con_holdout, feature_cols = prepare_splits(fm)
    print(f"\nSplits:")
    print(f"  Business train:  {X_pos_train.shape}")
    print(f"  Business holdout: {X_pos_holdout.shape}")
    print(f"  Consumer unlabeled: {X_unlabeled.shape}")
    print(f"  Consumer holdout:   {X_con_holdout.shape}")
    print(f"  Features used: {len(feature_cols)}")

    # ─── Baseline LogReg ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training Baseline LogReg")
    print("=" * 70)
    X_baseline_train = pd.concat([X_pos_train, X_unlabeled.sample(n=len(X_pos_train), random_state=RANDOM_STATE)])
    y_baseline_train = np.concatenate([np.ones(len(X_pos_train)), np.zeros(len(X_pos_train))])
    baseline = build_baseline(X_baseline_train, y_baseline_train)

    # Evaluate on holdout
    X_holdout = pd.concat([X_pos_holdout, X_con_holdout])
    y_holdout = np.concatenate([np.ones(len(X_pos_holdout)), np.zeros(len(X_con_holdout))])
    p_base = baseline.predict_proba(X_holdout)[:, 1]
    print(f"  Baseline ROC-AUC: {roc_auc_score(y_holdout, p_base):.4f}")
    print(f"  Baseline PR-AUC:  {average_precision_score(y_holdout, p_base):.4f}")
    save_model(baseline, "baseline_logreg")

    # ─── PU-Bagging LightGBM ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training PU-Bagging LightGBM")
    print("=" * 70)
    pu = PUBaggingClassifier(n_iterations=10, sample_size=20_000)
    pu.fit(X_pos_train, X_unlabeled, verbose=True)
    save_model(pu, "pu_bagging_lgbm")

    # Evaluate on holdout
    p_pu = pu.predict_proba_business(X_holdout)
    print(f"\n  PU-Bagging ROC-AUC: {roc_auc_score(y_holdout, p_pu):.4f}")
    print(f"  PU-Bagging PR-AUC:  {average_precision_score(y_holdout, p_pu):.4f}")
    print(f"  PU-Bagging F1@0.5:  {f1_score(y_holdout, p_pu > 0.5):.4f}")

    # Feature importances
    imp = pu.feature_importance(X_pos_train).head(15)
    print(f"\n  Top-15 features by importance:")
    for name, val in imp.items():
        print(f"    {name:<35} {val:>8.1f}")

    # ─── Score all 80K consumer ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Scoring all 80K consumer cards")
    print("=" * 70)
    consumer_fm = fm.filter(pl.col("label") == 0).to_pandas()
    X_consumer = consumer_fm[feature_cols]
    scores = pu.predict_proba_business(X_consumer)

    consumer_scored = consumer_fm.copy()
    consumer_scored["business_score"] = scores

    out = PROCESSED_DIR / "consumer_scored.parquet"
    pl.from_pandas(consumer_scored).write_parquet(out)
    print(f"  Saved: {out}")
    print(f"  Score stats: min={scores.min():.6f} max={scores.max():.4f} mean={scores.mean():.6f}")
    for q in [0.50, 0.90, 0.95, 0.99, 0.999, 1.0]:
        print(f"  q={q:.3f}: {np.quantile(scores, q):.6f}")
    print(f"  Cards > 0.5: {(scores > 0.5).sum()}")
    print(f"  Cards > 0.1: {(scores > 0.1).sum()}")
    print(f"  Cards > 0.01: {(scores > 0.01).sum()}")

    # ─── Submission.csv ───────────────────────────────────────────────
    sub = consumer_scored[["card_number", "business_score"]].rename(
        columns={"business_score": "score"}
    ).sort_values("score", ascending=False)
    sub.to_csv(ROOT / "submission.csv", index=False)
    print(f"\n  Submission saved: submission.csv ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
