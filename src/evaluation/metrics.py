"""
Метрики для PU-learning задачи:
  - confusion matrix, PR-AUC, ROC-AUC, F1 (на holdout)
  - Precision@K (бизнес-метрика)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    average_precision_score, f1_score,
    classification_report,
)

from src.config import DEFAULT_THRESHOLD


def evaluate_on_holdout(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    label: str = "",
) -> dict:
    """Полный набор метрик на holdout (business cards)."""
    y_pred = (y_score >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}ROC-AUC:  {roc_auc:.4f}")
    print(f"{prefix}PR-AUC:   {pr_auc:.4f}  ← основная метрика")
    print(f"{prefix}F1:       {f1:.4f}  (threshold={threshold})")
    print(f"\n{prefix}Confusion Matrix:")
    print(cm)
    print(f"\n{prefix}Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["consumer", "business"]))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Из top-K карт по скору — какая доля реально business."""
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return y_true[top_k_idx].mean()


def precision_at_k_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: list[int] | None = None,
) -> pd.DataFrame:
    if ks is None:
        ks = [100, 500, 1000, 2000, 5000]
    rows = [{"K": k, "Precision@K": precision_at_k(y_true, y_score, k)} for k in ks]
    return pd.DataFrame(rows)
