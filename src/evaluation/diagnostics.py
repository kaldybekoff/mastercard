"""
Diagnostics suite — формальная упаковка validation материалов для FINAL.ipynb.
Реализует рекомендации Q3 организаторов:
  1. Score distribution на Y (consumer)
  2. Top-N qualitative inspection
  3. Confusion Matrix на holdout
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ─── Score distribution ────────────────────────────────────────────────────


def score_distribution_stats(scores: np.ndarray) -> pd.DataFrame:
    """Перцентили + counts по разным порогам."""
    quantiles = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

    rows = []
    for q in quantiles:
        rows.append({
            "type": "quantile",
            "label": f"q={q:.3f}",
            "value": float(np.quantile(scores, q)),
        })
    for t in thresholds:
        rows.append({
            "type": "threshold",
            "label": f"n_above_{t}",
            "value": int((scores > t).sum()),
        })
    rows.append({"type": "summary", "label": "mean", "value": float(scores.mean())})
    rows.append({"type": "summary", "label": "median", "value": float(np.median(scores))})
    rows.append({"type": "summary", "label": "max", "value": float(scores.max())})

    return pd.DataFrame(rows)


def plot_score_distribution(
    scores: np.ndarray,
    business_scores: np.ndarray | None = None,
    save_path: str | None = None,
    title_suffix: str = "",
):
    """Двухпанельный график: linear hist + log-scale hist."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: linear histogram
    ax = axes[0]
    ax.hist(scores, bins=50, color="steelblue", alpha=0.7, label="Consumer (80K)")
    if business_scores is not None:
        ax.hist(business_scores, bins=50, color="darkorange", alpha=0.6,
                label=f"Business holdout ({len(business_scores):,})")
    ax.set_xlabel("Business Score")
    ax.set_ylabel("Number of cards")
    ax.set_title(f"Score distribution (linear){title_suffix}")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: log-scale Y to see tail
    ax = axes[1]
    bins = np.logspace(-6, 0, 60)
    ax.hist(scores, bins=bins, color="steelblue", alpha=0.7, label="Consumer (80K)")
    if business_scores is not None:
        ax.hist(business_scores, bins=bins, color="darkorange", alpha=0.6, label="Business holdout")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Business Score (log)")
    ax.set_ylabel("Number of cards (log)")
    ax.set_title(f"Score distribution (log-log) — shows the tail{title_suffix}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ─── Top-N qualitative inspection ──────────────────────────────────────────


KEY_BUSINESS_SIGNALS = [
    "merchant_hhi",          # Q4: концентрация по торговцам
    "b2b_spend_share",        # Q4: B2B MCC share
    "recurring_amount_share", # Q4: регулярные крупные списания
    "foreign_tx_share",       # Q4: трансграничные
    "business_merchant_overlap",
    "n_unique_merchants",
    "weekday_share",
    "evening_share",
]


def top_n_inspection(
    consumer_scored: pl.DataFrame,
    feature_matrix: pl.DataFrame,
    n_top: int = 50,
    score_col: str = "business_score",
) -> pd.DataFrame:
    """
    Возвращает сравнительную таблицу:
      каждая строка — фича Q4
      колонки: top-N medianа vs business median vs consumer median
    """
    biz = feature_matrix.filter(pl.col("label") == 1)
    con = feature_matrix.filter(pl.col("label") == 0)
    top_n = consumer_scored.sort(score_col, descending=True).head(n_top)

    rows = []
    for feat in KEY_BUSINESS_SIGNALS:
        if feat not in consumer_scored.columns:
            continue
        rows.append({
            "feature": feat,
            f"top{n_top}_median": float(top_n[feat].median()),
            "business_median": float(biz[feat].median()),
            "consumer_median": float(con[feat].median()),
            f"top{n_top}_vs_consumer": float(top_n[feat].median()) - float(con[feat].median()),
        })
    return pd.DataFrame(rows)


def top_n_detail(
    consumer_scored: pl.DataFrame,
    n_top: int = 20,
    score_col: str = "business_score",
) -> pd.DataFrame:
    """Подробная таблица: top-N карт с их сырыми значениями ключевых фичей."""
    cols = ["card_number", score_col] + [
        c for c in KEY_BUSINESS_SIGNALS if c in consumer_scored.columns
    ]
    return (
        consumer_scored.sort(score_col, descending=True)
        .head(n_top)
        .select(cols)
        .to_pandas()
    )


# ─── Confusion Matrix ──────────────────────────────────────────────────────


def confusion_matrix_holdout(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    save_path: str | None = None,
) -> tuple[np.ndarray, plt.Figure]:
    """Confusion matrix + интерпретация FP/FN cost."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Consumer", "Business"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (threshold={threshold})")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    interpretation = {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FP_cost": "Marketing spend on non-business (soft-touch, recoverable)",
        "FN_cost": "Missed hidden entrepreneur (lost LTV ~200K KZT)",
        "asymmetric": "FN is ~10x more costly — bank misses revenue opportunity",
    }
    return cm, fig, interpretation
