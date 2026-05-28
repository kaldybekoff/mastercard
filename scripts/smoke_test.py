"""
Smoke test: проверяет что весь pipeline собирается из артефактов.
НЕ выполняет тяжёлые операции — только проверяет что данные читаются,
модели загружаются, метрики совпадают с ожидаемыми.
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import polars as pl
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks = []

def check(name, passed, detail=""):
    status = "✓" if passed else "✗"
    checks.append((name, passed))
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")


print("=" * 70)
print("SMOKE TEST — End-to-end pipeline validation")
print("=" * 70)

# ─── 1. Raw data ──────────────────────────────────────────────────────
print("\n[1] Raw data files")
from src.config import BUSINESS_CARDS_PATH, CONSUMER_CARDS_PATH, MERCHANTS_PATH
check("business_cards exists", BUSINESS_CARDS_PATH.exists(), str(BUSINESS_CARDS_PATH.name))
check("consumer_cards exists", CONSUMER_CARDS_PATH.exists(), str(CONSUMER_CARDS_PATH.name))
check("merchants_reference exists", MERCHANTS_PATH.exists(), str(MERCHANTS_PATH.name))

# ─── 2. Feature matrix ────────────────────────────────────────────────
print("\n[2] Feature matrix")
from src.config import FEATURE_MATRIX_PATH
check("feature_matrix.parquet exists", FEATURE_MATRIX_PATH.exists())
if FEATURE_MATRIX_PATH.exists():
    fm = pl.read_parquet(FEATURE_MATRIX_PATH)
    check("feature_matrix has 105K rows", fm.shape[0] == 105_000, f"got {fm.shape[0]:,}")
    check("feature_matrix has 67 columns", fm.shape[1] == 67, f"got {fm.shape[1]}")
    check("no null values", sum(fm[c].null_count() for c in fm.columns) == 0)
    # Key bug-fix check
    biz_foreign = fm.filter(pl.col("label") == 1)["foreign_tx_share"].mean()
    check(
        "foreign_tx_share is NOT constant (bug fix applied)",
        0.1 < biz_foreign < 0.5,
        f"business mean foreign_tx_share = {biz_foreign:.3f}",
    )

# ─── 3. Models ─────────────────────────────────────────────────────────
print("\n[3] Trained models")
from src.config import MODELS_DIR
baseline_path = MODELS_DIR / "baseline_logreg.pkl"
pu_path = MODELS_DIR / "pu_bagging_lgbm.pkl"
check("baseline_logreg.pkl exists", baseline_path.exists())
check("pu_bagging_lgbm.pkl exists", pu_path.exists())
if pu_path.exists():
    pu_model = joblib.load(pu_path)
    check("PU-Bagging has 10 sub-models", len(pu_model.models_) == 10, f"got {len(pu_model.models_)}")

# ─── 4. Scored consumer ───────────────────────────────────────────────
print("\n[4] Scored consumer cards")
from src.config import PROCESSED_DIR
scored_path = PROCESSED_DIR / "consumer_scored.parquet"
scored_v2_path = PROCESSED_DIR / "consumer_scored_v2.parquet"
check("consumer_scored.parquet exists", scored_path.exists())
check("consumer_scored_v2.parquet exists (with anomaly)", scored_v2_path.exists())
if scored_path.exists():
    scored = pl.read_parquet(scored_path)
    check("has 80K rows", scored.shape[0] == 80_000, f"got {scored.shape[0]:,}")
    check("has business_score column", "business_score" in scored.columns)
    max_score = float(scored["business_score"].max())
    check("max business_score > 0.8", max_score > 0.8, f"max = {max_score:.4f}")
if scored_v2_path.exists():
    scored_v2 = pl.read_parquet(scored_v2_path)
    check("v2 has combined_score column", "combined_score" in scored_v2.columns)

# ─── 5. Submission files ──────────────────────────────────────────────
print("\n[5] Submission files")
sub_path = ROOT / "submission.csv"
sub_comb_path = ROOT / "submission_combined.csv"
check("submission.csv exists", sub_path.exists())
check("submission_combined.csv exists", sub_comb_path.exists())
if sub_path.exists():
    sub = pl.read_csv(sub_path)
    check("submission has 80K rows", sub.shape[0] == 80_000)
    check("submission has card_number and score", set(sub.columns) == {"card_number", "score"})
    is_sorted = (sub["score"].to_numpy()[:-1] >= sub["score"].to_numpy()[1:]).all()
    check("submission sorted by score desc", is_sorted)
    check("no null scores", sub["score"].null_count() == 0)

# ─── 6. Reports / diagnostics ─────────────────────────────────────────
print("\n[6] Diagnostic reports")
diag_dir = ROOT / "reports" / "diagnostics"
required_files = [
    "cv_results.csv", "cv_summary.csv", "holdout_metrics.csv",
    "score_distribution.csv", "score_distribution.png",
    "top50_inspection.csv", "top20_detail.csv",
    "confusion_matrix.png", "feature_importance.csv",
    "shap_global.png", "shap_top_features.csv",
    "ablation_results.csv",
    "archetypes.md", "archetypes.json",
    "pca_2d_visualization.png", "pca_2d_zoom.png",
    "segments_radar.png", "segment_profiles.csv",
]
for fname in required_files:
    check(f"{fname}", (diag_dir / fname).exists())

# ─── 7. Source modules importable ─────────────────────────────────────
print("\n[7] Source modules import OK")
try:
    from src.features.build_features import build_feature_matrix
    check("build_features.py", True)
except Exception as e:
    check("build_features.py", False, str(e))

try:
    from src.models.pu_bagging import PUBaggingClassifier
    from src.models.baseline import build_baseline
    from src.models.train import prepare_splits
    check("models modules", True)
except Exception as e:
    check("models modules", False, str(e))

try:
    from src.evaluation.cv import run_5fold_cv, summarize_cv
    from src.evaluation.diagnostics import (
        score_distribution_stats, plot_score_distribution,
        top_n_inspection, top_n_detail, confusion_matrix_holdout,
    )
    from src.evaluation.shap_analysis import build_explainer, shap_global_summary
    check("evaluation modules", True)
except Exception as e:
    check("evaluation modules", False, str(e))

try:
    from src.segmentation.cluster import fit_kmeans, build_segment_profiles
    check("segmentation modules", True)
except Exception as e:
    check("segmentation modules", False, str(e))

# ─── 8. CV results numeric check ──────────────────────────────────────
print("\n[8] CV results numeric check")
cv_csv = diag_dir / "cv_summary.csv"
if cv_csv.exists():
    cv = pd.read_csv(cv_csv)
    roc_row = cv[cv["metric"] == "roc_auc"].iloc[0]
    roc_mean = float(roc_row["mean"])
    check(
        "CV ROC-AUC ≥ 0.9999",
        roc_mean >= 0.9999,
        f"mean = {roc_mean:.6f}",
    )

# ─── 9. Ablation results ──────────────────────────────────────────────
print("\n[9] Ablation results check")
abl_csv = diag_dir / "ablation_results.csv"
if abl_csv.exists():
    abl = pd.read_csv(abl_csv)
    only_mcc = abl[abl["experiment"] == "Only MCC-features (Group C)"]
    if len(only_mcc) > 0:
        roc = float(only_mcc.iloc[0]["roc_auc_mean"])
        check("Only-MCC ablation ROC-AUC > 0.99", roc > 0.99, f"got {roc:.4f}")

# ─── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
total = len(checks)
passed = sum(1 for _, ok in checks if ok)
print(f"RESULT: {passed} / {total} checks passed")
if passed == total:
    print("✅ ALL CHECKS PASSED — pipeline is ready for submission")
else:
    print("❌ SOME CHECKS FAILED")
    for name, ok in checks:
        if not ok:
            print(f"   FAILED: {name}")
print("=" * 70)
sys.exit(0 if passed == total else 1)
