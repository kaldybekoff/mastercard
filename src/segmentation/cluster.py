"""
Segmentation of hidden entrepreneurs: KMeans + segment profiling.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import RANDOM_STATE, N_SEGMENTS, SEGMENT_NAMES


def fit_kmeans(
    X: pd.DataFrame,
    n_clusters: int = N_SEGMENTS,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Fit KMeans on scaled features. Returns (model, labels, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(X_scaled)
    return model, labels, scaler


def build_segment_profiles(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list,
) -> pd.DataFrame:
    """Compute per-segment medians for selected features."""
    df = X[feature_cols].copy()
    df["segment"] = labels
    profiles = (
        df.groupby("segment")[feature_cols]
        .median()
        .round(4)
    )
    profiles.index = [SEGMENT_NAMES.get(i, f"Segment {i}") for i in profiles.index]
    return profiles


def plot_segment_radar(
    profiles: pd.DataFrame,
    features: list,
    title: str = "Segment Profiles (median, normalised)",
) -> plt.Figure:
    """Radar/spider chart comparing segment profiles."""
    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))

    norm_profiles = profiles[features].copy()
    for col in features:
        vmin, vmax = norm_profiles[col].min(), norm_profiles[col].max()
        norm_profiles[col] = (norm_profiles[col] - vmin) / (vmax - vmin) if vmax > vmin else 0.5

    for (seg_name, row), color in zip(norm_profiles.iterrows(), colors):
        values = row[features].tolist() + row[features].tolist()[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=seg_name)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    return fig


def plot_segment_bars(
    profiles: pd.DataFrame,
    features: list,
    title: str = "Segment Profiles — median feature values",
) -> plt.Figure:
    """Grouped bar chart: bars = segments, groups = features."""
    n_segs = len(profiles)
    n_feat = len(features)
    x = np.arange(n_feat)
    width = 0.8 / n_segs
    colors = plt.cm.tab10(np.linspace(0, 1, n_segs))

    fig, ax = plt.subplots(figsize=(max(12, n_feat * 1.2), 5))
    for i, (seg_name, row) in enumerate(profiles[features].iterrows()):
        offset = (i - n_segs / 2 + 0.5) * width
        ax.bar(x + offset, row[features].values, width, label=seg_name,
               color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Median value")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pca_scatter(
    X: pd.DataFrame,
    labels: np.ndarray,
    scaler: StandardScaler,
    title: str = "PCA — Hidden Entrepreneurs by Segment",
) -> plt.Figure:
    """2D PCA scatter coloured by segment."""
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    colors = plt.cm.tab10(np.linspace(0, 1, N_SEGMENTS))
    fig, ax = plt.subplots(figsize=(9, 7))
    for seg_id, seg_name in SEGMENT_NAMES.items():
        mask = labels == seg_id
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.5,
                   color=colors[seg_id], label=f"{seg_name} (n={mask.sum():,})")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig
