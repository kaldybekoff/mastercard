"""
Phase 2: Validation suite.
Запускает все рекомендованные Q3 валидации и сохраняет артефакты для FINAL.ipynb.

Output:
  reports/diagnostics/cv_results.csv        — 5-fold CV per-fold
  reports/diagnostics/cv_summary.csv         — mean ± std
  reports/diagnostics/score_distribution.csv — перцентили + counts
  reports/diagnostics/score_distribution.png — гистограмма
  reports/diagnostics/top50_inspection.csv   — сравнение медиан
  reports/diagnostics/top20_detail.csv       — детали топ-20 карт
  reports/diagnostics/confusion_matrix.png   — CM на holdout
  reports/diagnostics/holdout_metrics.csv    — ROC-AUC, PR-AUC, F1 на holdout
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import polars as pl
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_MATRIX_PATH, MODELS_DIR, PROCESSED_DIR
from src.models.train import prepare_splits
from src.evaluation.cv import run_5fold_cv, summarize_cv
from src.evaluation.diagnostics import (
    score_distribution_stats, plot_score_distribution,
    top_n_inspection, top_n_detail,
    confusion_matrix_holdout,
)

OUT_DIR = ROOT / "reports" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("PHASE 2: VALIDATION SUITE")
    print("=" * 70)

    # ─── Data loading ──────────────────────────────────────────────
    fm = pl.read_parquet(FEATURE_MATRIX_PATH)
    consumer_scored = pl.read_parquet(PROCESSED_DIR / "consumer_scored.parquet")
    pu_model = joblib.load(MODELS_DIR / "pu_bagging_lgbm.pkl")
    X_pos_train, X_pos_holdout, X_unlabeled, X_con_holdout, feature_cols = prepare_splits(fm)

    # ─── 2a. 5-fold CV ────────────────────────────────────────────
    print("\n[2a] Running 5-fold StratifiedKFold CV on Dataset X")
    cv_df = run_5fold_cv(X_pos_train, X_unlabeled, n_folds=5)
    cv_summary = summarize_cv(cv_df)

    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)
    cv_summary.to_csv(OUT_DIR / "cv_summary.csv", index=False)

    print("\nCV per-fold:")
    print(cv_df.to_string(index=False))
    print("\nCV summary (mean ± std):")
    print(cv_summary.to_string(index=False))

    # ─── 2b. Confusion Matrix + Holdout metrics ────────────────────
    print("\n[2b] Confusion Matrix on holdout")
    import pandas as _pd
    X_holdout = _pd.concat([X_pos_holdout, X_con_holdout])
    y_holdout = np.concatenate([np.ones(len(X_pos_holdout)), np.zeros(len(X_con_holdout))])
    p_holdout = pu_model.predict_proba_business(X_holdout)

    holdout_metrics = pd.DataFrame([
        {"metric": "roc_auc", "value": roc_auc_score(y_holdout, p_holdout)},
        {"metric": "pr_auc", "value": average_precision_score(y_holdout, p_holdout)},
        {"metric": "f1_at_0.5", "value": f1_score(y_holdout, p_holdout > 0.5)},
        {"metric": "precision_at_0.5", "value": precision_score(y_holdout, p_holdout > 0.5, zero_division=0)},
        {"metric": "recall_at_0.5", "value": recall_score(y_holdout, p_holdout > 0.5)},
    ])
    holdout_metrics.to_csv(OUT_DIR / "holdout_metrics.csv", index=False)
    print(holdout_metrics.to_string(index=False))

    cm, cm_fig, cm_interp = confusion_matrix_holdout(
        y_holdout, p_holdout, threshold=0.5,
        save_path=str(OUT_DIR / "confusion_matrix.png"),
    )
    print(f"\nCM: TN={cm_interp['TN']:,}  FP={cm_interp['FP']:,}  FN={cm_interp['FN']:,}  TP={cm_interp['TP']:,}")
    print(f"  FP cost: {cm_interp['FP_cost']}")
    print(f"  FN cost: {cm_interp['FN_cost']}")
    print(f"  Asymmetry: {cm_interp['asymmetric']}")

    # ─── 2c. Score Distribution ────────────────────────────────────
    print("\n[2c] Score Distribution Analysis")
    scores_consumer = consumer_scored["business_score"].to_numpy()
    scores_business_holdout = p_holdout[: len(X_pos_holdout)]

    dist_stats = score_distribution_stats(scores_consumer)
    dist_stats.to_csv(OUT_DIR / "score_distribution.csv", index=False)
    print(dist_stats.to_string(index=False))

    plot_score_distribution(
        scores_consumer,
        business_scores=scores_business_holdout,
        save_path=str(OUT_DIR / "score_distribution.png"),
        title_suffix=" — Consumer (Y) vs Business holdout",
    )

    # ─── 2d. Top-N inspection ──────────────────────────────────────
    print("\n[2d] Top-50 Qualitative Inspection")
    top50_table = top_n_inspection(consumer_scored, fm, n_top=50)
    top50_table.to_csv(OUT_DIR / "top50_inspection.csv", index=False)
    print(top50_table.to_string(index=False))

    top20_detail = top_n_detail(consumer_scored, n_top=20)
    top20_detail.to_csv(OUT_DIR / "top20_detail.csv", index=False)
    print(f"\nTop-20 detail saved to top20_detail.csv ({len(top20_detail)} rows)")

    # ─── Feature importance ────────────────────────────────────────
    print("\n[2e] Feature Importance (from trained model)")
    imp = pu_model.feature_importance(X_pos_train)
    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(imp_df.head(20).to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"All artifacts saved to: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
