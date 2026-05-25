"""
Стандартные графики для ноутбука и презентации.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
)


def plot_pr_curve(y_true: np.ndarray, scores: dict[str, np.ndarray], ax=None):
    """PR-кривые для нескольких моделей. scores = {'LightGBM': arr, ...}"""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, y_score in scores.items():
        p, r, _ = precision_recall_curve(y_true, y_score)
        from sklearn.metrics import average_precision_score
        auc = average_precision_score(y_true, y_score)
        ax.plot(r, p, label=f"{name} (AP={auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_roc_curve(y_true: np.ndarray, scores: dict[str, np.ndarray], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["pred: consumer", "pred: business"],
        yticklabels=["true: consumer", "true: business"],
        ax=ax,
    )
    ax.set_title(title)
    return ax


def plot_score_distribution(
    scores_biz: np.ndarray,
    scores_con: np.ndarray,
    threshold: float,
    ax=None,
):
    """Распределение business-score для известных business vs consumer."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores_biz, bins=50, alpha=0.6, label="Business (holdout)", color="#636EFA")
    ax.hist(scores_con, bins=50, alpha=0.6, label="Consumer", color="#EF553B")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold}")
    ax.set_xlabel("Business Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_precision_at_k(precision_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(precision_df["K"], precision_df["Precision@K"], marker="o")
    ax.set_xlabel("K (top-K cards)")
    ax.set_ylabel("Precision@K")
    ax.set_title("Precision@K — Business cards in top-K")
    ax.grid(alpha=0.3)
    return ax


def plot_feature_importance(importances: pd.Series, top_n: int = 20, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    top = importances.head(top_n)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="Blues_r")
    ax.set_title(f"Top-{top_n} Feature Importances (LightGBM)")
    ax.set_xlabel("Mean importance across PU iterations")
    ax.grid(alpha=0.3, axis="x")
    return ax
