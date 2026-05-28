"""
Cross-validation suite — рекомендовано Q3 организаторов.
5-fold StratifiedKFold на Dataset X (business + sampled consumer)
с метриками: ROC-AUC (главная), PR-AUC, F1, Precision, Recall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.config import RANDOM_STATE
from src.models.pu_bagging import DEFAULT_LGB_PARAMS


def run_5fold_cv(
    X_pos: pd.DataFrame,
    X_unlabeled: pd.DataFrame,
    n_folds: int = 5,
    threshold: float = 0.5,
    lgb_params: dict | None = None,
) -> pd.DataFrame:
    """
    Каждый fold:
      1. Train: 80% business + same-size sample из unlabeled (как negative)
      2. Test:  20% business + same-size sample из *other* unlabeled
      3. Считаем все метрики

    Возвращает DataFrame: fold × metric.
    """
    lgb_params = lgb_params or DEFAULT_LGB_PARAMS
    rng = np.random.default_rng(RANDOM_STATE)

    n_pos = len(X_pos)
    n_unl = len(X_unlabeled)
    sample_size = min(n_pos, n_unl // 2)  # train_neg + test_neg should fit

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    y_pos_dummy = np.ones(n_pos)

    results = []
    for fold_idx, (pos_train_idx, pos_test_idx) in enumerate(skf.split(X_pos, y_pos_dummy)):
        # Sample 2x batches of unlabeled — one for train_neg, one for test_neg
        n_train_neg = len(pos_train_idx)
        n_test_neg = len(pos_test_idx)

        all_unl_idx = rng.permutation(n_unl)
        train_neg_idx = all_unl_idx[:n_train_neg]
        test_neg_idx = all_unl_idx[n_train_neg : n_train_neg + n_test_neg]

        X_train = pd.concat(
            [X_pos.iloc[pos_train_idx], X_unlabeled.iloc[train_neg_idx]],
            ignore_index=True,
        )
        y_train = np.concatenate([np.ones(n_train_neg), np.zeros(n_train_neg)])

        X_test = pd.concat(
            [X_pos.iloc[pos_test_idx], X_unlabeled.iloc[test_neg_idx]],
            ignore_index=True,
        )
        y_test = np.concatenate([np.ones(n_test_neg), np.zeros(n_test_neg)])

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results.append({
            "fold": fold_idx + 1,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    df = pd.DataFrame(results)
    return df


def summarize_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Аггрегат mean ± std по каждой метрике."""
    metric_cols = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    rows = []
    for m in metric_cols:
        rows.append({
            "metric": m,
            "mean": cv_df[m].mean(),
            "std": cv_df[m].std(),
            "min": cv_df[m].min(),
            "max": cv_df[m].max(),
        })
    return pd.DataFrame(rows)
