"""
Phase 2f: SHAP explainability.

Output:
  reports/diagnostics/shap_global.png
  reports/diagnostics/shap_local_high.png    — карта с высоким score
  reports/diagnostics/shap_local_mid.png     — карта со средним score
  reports/diagnostics/shap_local_low.png     — карта с нулевым score
  reports/diagnostics/shap_top_features.csv  — топ-фичи по mean |SHAP|
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_MATRIX_PATH, MODELS_DIR, PROCESSED_DIR
from src.models.train import prepare_splits
from src.evaluation.shap_analysis import build_explainer, shap_global_summary, shap_local_waterfall

OUT_DIR = ROOT / "reports" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("PHASE 2f: SHAP EXPLAINABILITY")
    print("=" * 70)

    fm = pl.read_parquet(FEATURE_MATRIX_PATH)
    consumer_scored = pl.read_parquet(PROCESSED_DIR / "consumer_scored.parquet")
    pu_model = joblib.load(MODELS_DIR / "pu_bagging_lgbm.pkl")
    X_pos_train, _, X_unlabeled, _, feature_cols = prepare_splits(fm)

    print("\nBuilding TreeExplainer on first LGB model...")
    explainer = build_explainer(pu_model, X_pos_train)

    # ─── Global summary ──────────────────────────────────────────
    print("\n[Global] computing SHAP on 2000-card sample (mix of business + consumer)")
    import pandas as _pd
    sample_X = _pd.concat([
        X_pos_train.sample(n=1000, random_state=42),
        X_unlabeled.sample(n=1000, random_state=42),
    ]).reset_index(drop=True)

    fig_global, shap_vals = shap_global_summary(
        explainer, sample_X, n_sample=2000,
        save_path=str(OUT_DIR / "shap_global.png"),
    )
    plt.close(fig_global)

    # Top features by mean |SHAP|
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_features = (
        pd.DataFrame({"feature": sample_X.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top_features.to_csv(OUT_DIR / "shap_top_features.csv", index=False)
    print("\nTop-15 features by mean |SHAP|:")
    print(top_features.head(15).to_string(index=False))

    # ─── Local: 3 example consumer cards ─────────────────────────
    print("\n[Local] 3 example consumer cards: high / mid / low scoring")

    scored_pd = consumer_scored.to_pandas()
    scored_pd = scored_pd.sort_values("business_score", ascending=False).reset_index(drop=True)

    # High: top-1
    high_idx = 0
    # Mid: card with score ~ q=0.99 (rank ~800)
    mid_idx = 800
    # Low: median card (score ~ 1e-6)
    low_idx = 40000

    for label, idx, save_name in [
        ("HIGH score (top-1)", high_idx, "shap_local_high.png"),
        ("MID score (rank ~800)", mid_idx, "shap_local_mid.png"),
        ("LOW score (median)", low_idx, "shap_local_low.png"),
    ]:
        row = scored_pd.iloc[[idx]][feature_cols].reset_index(drop=True)
        score = scored_pd.iloc[idx]["business_score"]
        card = scored_pd.iloc[idx]["card_number"]
        title = f"{label}  |  card={card}  |  score={score:.6f}"
        print(f"  {title}")
        fig = shap_local_waterfall(
            explainer, row, title=title,
            save_path=str(OUT_DIR / save_name),
        )
        plt.close(fig)

    print(f"\n{'=' * 70}")
    print(f"All SHAP artifacts saved to: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
