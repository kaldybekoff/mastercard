"""
Synthetic Injection Test — уникальная валидация recall для PU-learning.

Идея: берём N business-карт из holdout, убираем их метку, добавляем в consumer pool.
Затем запускаем inference и проверяем, какая доля injected-карт попала в top-K%.
"""
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE


def run_injection_test(
    pu_model,
    X_pos_holdout: pd.DataFrame,
    X_unlabeled: pd.DataFrame,
    n_inject: int = 1000,
    top_pct: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Args:
        pu_model: обученный PUBaggingClassifier (уже fit)
        X_pos_holdout: holdout business cards (не участвовали в обучении)
        X_unlabeled: consumer cards
        n_inject: сколько holdout-карт вколоть в consumer pool
        top_pct: порог — карта "найдена" если попала в top X% по скору

    Returns:
        dict с recall и другими метриками
    """
    rng = np.random.default_rng(RANDOM_STATE)
    inject_idx = rng.choice(len(X_pos_holdout), size=min(n_inject, len(X_pos_holdout)), replace=False)
    X_inject = X_pos_holdout.iloc[inject_idx]

    # Pool = injected business + all consumer
    X_pool = pd.concat([X_inject, X_unlabeled], ignore_index=True)
    inject_flags = np.zeros(len(X_pool), dtype=bool)
    inject_flags[:len(X_inject)] = True

    # Inference
    scores = pu_model.predict_proba_business(X_pool)

    # Top-K% threshold
    k = max(1, int(len(X_pool) * top_pct))
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_mask = np.zeros(len(X_pool), dtype=bool)
    top_k_mask[top_k_idx] = True

    n_found = inject_flags[top_k_mask].sum()
    recall = n_found / len(X_inject)

    if verbose:
        print(f"Synthetic Injection Test:")
        print(f"  Injected: {len(X_inject)} business cards into {len(X_unlabeled)} consumer pool")
        print(f"  Pool size: {len(X_pool):,}")
        print(f"  Top-{top_pct*100:.0f}% = {k:,} cards")
        print(f"  Found:    {n_found} / {len(X_inject)}  →  Recall = {recall:.3f}")

    return {
        "n_inject": len(X_inject),
        "n_pool": len(X_pool),
        "top_k": k,
        "top_pct": top_pct,
        "n_found": int(n_found),
        "recall": recall,
        "scores": scores,
        "inject_flags": inject_flags,
    }
