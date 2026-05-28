"""
SHAP explainability — критерий 7 (Documentation & Explainability, 5%).

Берём среднюю модель из PU-Bagging (первый LGB из ансамбля как репрезентативный),
строим TreeExplainer, считаем global importance + 3 local examples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb


def build_explainer(pu_model, X_sample: pd.DataFrame) -> shap.TreeExplainer:
    """Берём первую LGB модель из ансамбля — SHAP values стабильны между итерациями."""
    base_model: lgb.LGBMClassifier = pu_model.models_[0]
    return shap.TreeExplainer(base_model)


def shap_global_summary(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    n_sample: int = 2000,
    save_path: str | None = None,
):
    """Beeswarm plot топ-фичей по среднему |SHAP|."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_sub = X.iloc[idx]
    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # class 1

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sub, max_display=20, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Global Feature Importance", y=1.02, fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig, shap_values


def shap_local_waterfall(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    title: str = "",
    save_path: str | None = None,
):
    """Waterfall для одной карты — объясняет почему она получила свой score."""
    shap_values = explainer.shap_values(X_row)
    expected = explainer.expected_value
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
        expected = expected[1] if hasattr(expected, "__len__") else expected
    sv = shap_values[0] if shap_values.ndim == 2 else shap_values

    explanation = shap.Explanation(
        values=sv,
        base_values=expected,
        data=X_row.iloc[0].values,
        feature_names=list(X_row.columns),
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
