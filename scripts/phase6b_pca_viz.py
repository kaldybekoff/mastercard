"""
Phase 6b: 2D PCA visualization — облака business / consumer / hidden entrepreneurs.

Цель: визуально показать что hidden entrepreneurs формируют чёткое облако
между типичным business и типичным consumer.
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_MATRIX_PATH, PROCESSED_DIR, RANDOM_STATE
from src.models.train import prepare_splits

OUT_DIR = ROOT / "reports" / "diagnostics"


def main():
    print("=" * 70)
    print("PHASE 6b: 2D PCA VISUALIZATION")
    print("=" * 70)

    fm = pl.read_parquet(FEATURE_MATRIX_PATH)
    scored = pl.read_parquet(PROCESSED_DIR / "consumer_scored_v2.parquet")
    _, _, _, _, feature_cols = prepare_splits(fm)

    # Extract feature subsets
    business = fm.filter(pl.col("label") == 1).to_pandas()[feature_cols]
    consumer = fm.filter(pl.col("label") == 0).to_pandas()[feature_cols]

    # Top-N hidden entrepreneurs by combined score
    scored_pd = scored.to_pandas().sort_values("combined_score", ascending=False)
    top_card_numbers = scored_pd.head(165)["card_number"].values  # 165 = cards with PU > 0.001
    consumer_full = fm.filter(pl.col("label") == 0).to_pandas()
    hidden_mask = consumer_full["card_number"].isin(top_card_numbers)
    hidden = consumer_full[hidden_mask][feature_cols]

    # The "typical" consumer = everyone NOT in top-N
    typical_consumer = consumer_full[~hidden_mask][feature_cols]

    print(f"\nBusiness:        {len(business):,}")
    print(f"Hidden (top-165): {len(hidden):,}")
    print(f"Typical consumer: {len(typical_consumer):,}")

    # ─── Combine + fit PCA ────────────────────────────────────────
    # For balanced visualization, sample consumer
    rng = np.random.default_rng(RANDOM_STATE)
    typical_sample = typical_consumer.sample(n=5000, random_state=RANDOM_STATE)
    biz_sample = business.sample(n=min(5000, len(business)), random_state=RANDOM_STATE)

    all_X = np.vstack([
        biz_sample.values,
        typical_sample.values,
        hidden.values,
    ])
    labels = (
        ["Business"] * len(biz_sample)
        + ["Typical Consumer"] * len(typical_sample)
        + ["Hidden Entrepreneur"] * len(hidden)
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}")
    print(f"Combined: {sum(var_explained):.3f}")

    # ─── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))

    color_map = {
        "Business": "#1f77b4",         # blue
        "Typical Consumer": "#bbbbbb",  # gray
        "Hidden Entrepreneur": "#d62728",  # red
    }
    marker_map = {
        "Business": "o",
        "Typical Consumer": ".",
        "Hidden Entrepreneur": "X",
    }
    size_map = {
        "Business": 8,
        "Typical Consumer": 3,
        "Hidden Entrepreneur": 60,
    }
    alpha_map = {
        "Business": 0.35,
        "Typical Consumer": 0.20,
        "Hidden Entrepreneur": 0.85,
    }

    # Plot in order: consumer (background), business (middle), hidden (foreground)
    for group in ["Typical Consumer", "Business", "Hidden Entrepreneur"]:
        mask = np.array(labels) == group
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color_map[group],
            marker=marker_map[group],
            s=size_map[group],
            alpha=alpha_map[group],
            label=f"{group} (n={mask.sum():,})",
            edgecolors="white" if group == "Hidden Entrepreneur" else "none",
            linewidths=0.5,
        )

    ax.set_xlabel(f"PC1  ({var_explained[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var_explained[1]:.1%} variance)", fontsize=12)
    ax.set_title(
        "2D Projection of card-level feature space\n"
        "Hidden entrepreneurs cluster inside business region",
        fontsize=13, pad=15,
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_facecolor("#f8f8f8")

    plt.tight_layout()
    save_path = OUT_DIR / "pca_2d_visualization.png"
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {save_path}")

    # ─── Also: 2D zoom into top-N + business region ───────────────
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    # Plot only business + hidden, no consumer
    biz_mask = np.array(labels) == "Business"
    hid_mask = np.array(labels) == "Hidden Entrepreneur"
    ax2.scatter(X_pca[biz_mask, 0], X_pca[biz_mask, 1],
                c="#1f77b4", marker="o", s=10, alpha=0.4,
                label=f"Business (n={biz_mask.sum():,})")
    ax2.scatter(X_pca[hid_mask, 0], X_pca[hid_mask, 1],
                c="#d62728", marker="X", s=80, alpha=0.85,
                edgecolors="white", linewidths=0.8,
                label=f"Hidden Entrepreneur (n={hid_mask.sum():,})")
    ax2.set_xlabel(f"PC1  ({var_explained[0]:.1%} variance)", fontsize=12)
    ax2.set_ylabel(f"PC2  ({var_explained[1]:.1%} variance)", fontsize=12)
    ax2.set_title("Zoom: Business vs Hidden Entrepreneurs only", fontsize=13, pad=15)
    ax2.legend(loc="best", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_facecolor("#f8f8f8")
    plt.tight_layout()
    save_path2 = OUT_DIR / "pca_2d_zoom.png"
    plt.savefig(save_path2, dpi=140, bbox_inches="tight")
    print(f"Saved: {save_path2}")


if __name__ == "__main__":
    main()
