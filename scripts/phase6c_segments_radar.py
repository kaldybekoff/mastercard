"""
Phase 6c: Segment radar charts.

K-Means на топ-1000 cards (по combined score), 5 сегментов,
радар-чарты с интерпретируемыми бизнес-фичами.
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, RANDOM_STATE

OUT_DIR = ROOT / "reports" / "diagnostics"

# Бизнес-интерпретируемые фичи для сегментации
SEG_FEATURES = [
    "b2b_spend_share",
    "tokenized_share",
    "online_share",
    "rental_tx_share",
    "merchant_hhi",
    "recurring_amount_share",
    "foreign_tx_share",
    "weekday_share",
]

# Понятные подписи для радар-чарта
FEAT_LABELS = {
    "b2b_spend_share":        "B2B spend",
    "tokenized_share":        "Apple/Google Pay",
    "online_share":           "Online",
    "rental_tx_share":        "Rental MCC",
    "merchant_hhi":           "Concentration",
    "recurring_amount_share": "Recurring",
    "foreign_tx_share":       "Foreign tx",
    "weekday_share":          "Weekday",
}


def main():
    print("=" * 70)
    print("PHASE 6c: SEGMENT RADAR CHARTS")
    print("=" * 70)

    scored = pl.read_parquet(PROCESSED_DIR / "consumer_scored_v2.parquet")
    top = scored.sort("combined_score", descending=True).head(1000).to_pandas()

    print(f"\nClustering top-{len(top)} cards into 5 segments")

    X_seg = top[SEG_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    km = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    seg_labels = km.fit_predict(X_scaled)
    top["segment"] = seg_labels

    # Median profile per segment
    profiles = top.groupby("segment")[SEG_FEATURES].median()
    counts = top.groupby("segment").size().rename("n_cards")
    print(f"\nSegment profiles (median):")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(profiles.join(counts).to_string())

    # ─── Назначим имена сегментам по доминирующей фиче ────────────
    segment_names = {}
    used_names = set()
    # Sort segments by most distinctive feature to assign unique names
    for seg_id, row in profiles.iterrows():
        if row["rental_tx_share"] > 0.02:
            name = "Rental / Hospitality"
        elif row["b2b_spend_share"] < 0.3 and row["merchant_hhi"] > 0.5:
            # Concentrated but NOT on B2B MCCs — likely specialty consumer service
            name = "Specialty Service Provider"
        elif row["recurring_amount_share"] > 0.2 and row["foreign_tx_share"] > 0.35:
            name = "Digital / SaaS Operator"
        elif row["merchant_hhi"] > 0.75 and row["b2b_spend_share"] > 0.9:
            name = "Wholesale Trader"
        elif row["online_share"] < 0.75 or row["merchant_hhi"] < 0.55:
            # Less concentrated, partly offline — diversified small business
            name = "Diversified Small Business"
        elif row["weekday_share"] > 0.85 and row["b2b_spend_share"] > 0.5:
            name = "Office-hours B2B"
        else:
            name = f"Mixed B2B #{seg_id}"

        # Disambiguate duplicates
        base = name
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base} ({suffix})"
        used_names.add(name)
        segment_names[seg_id] = name

    print(f"\nSegment names:")
    for sid, name in segment_names.items():
        print(f"  {sid}: {name} (n={counts[sid]})")

    # ─── Radar chart helper ────────────────────────────────────────
    n_features = len(SEG_FEATURES)
    angles = [n / float(n_features) * 2 * np.pi for n in range(n_features)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5.5), subplot_kw=dict(polar=True))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Normalize each feature to [0,1] across all segments for visual comparison
    # (radar chart works best with values in similar range)
    feat_min = profiles.min()
    feat_max = profiles.max()
    profiles_norm = (profiles - feat_min) / (feat_max - feat_min + 1e-9)

    for i, (seg_id, ax) in enumerate(zip(sorted(profiles_norm.index), axes)):
        values = profiles_norm.loc[seg_id, SEG_FEATURES].tolist()
        values += values[:1]  # close the loop

        color = palette[i % len(palette)]
        ax.fill(angles, values, color=color, alpha=0.30)
        ax.plot(angles, values, color=color, linewidth=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([FEAT_LABELS[f] for f in SEG_FEATURES], fontsize=9)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels([], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{segment_names[seg_id]}\n({counts[seg_id]} cards)",
            fontsize=11, pad=15, fontweight="bold",
        )

    fig.suptitle(
        "Hidden Entrepreneur Segments — KMeans (k=5) on Top-1000 cards",
        fontsize=14, y=1.02, fontweight="bold",
    )
    plt.tight_layout()
    save_path = OUT_DIR / "segments_radar.png"
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {save_path}")

    # ─── Save segment profile CSV ──────────────────────────────────
    profile_table = profiles.copy()
    profile_table["n_cards"] = counts
    profile_table["segment_name"] = [segment_names[s] for s in profile_table.index]
    cols_order = ["segment_name", "n_cards"] + SEG_FEATURES
    profile_table[cols_order].to_csv(OUT_DIR / "segment_profiles.csv")
    print(f"Saved: {OUT_DIR / 'segment_profiles.csv'}")


if __name__ == "__main__":
    main()
