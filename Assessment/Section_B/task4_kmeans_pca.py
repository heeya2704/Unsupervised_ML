"""
Section B — Task 4: Customer Segmentation with K-Means and PCA Visualisation
M7-A1 | StandardScaler, Elbow Method (k=2..9), final K-Means, PCA scatter
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_CUSTOMERS = 180
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def make_customers(n: int = N_CUSTOMERS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)

    casual = np.column_stack(
        [
            rng.integers(1, 5, size=n1),
            rng.normal(180, 35, size=n1),
            rng.normal(3.6, 0.35, size=n1),
        ]
    )
    regular = np.column_stack(
        [
            rng.integers(6, 14, size=n2),
            rng.normal(320, 40, size=n2),
            rng.normal(4.2, 0.25, size=n2),
        ]
    )
    premium = np.column_stack(
        [
            rng.integers(15, 28, size=n3),
            rng.normal(620, 55, size=n3),
            rng.normal(4.7, 0.18, size=n3),
        ]
    )

    data = np.vstack([casual, regular, premium])
    rng.shuffle(data)

    df = pd.DataFrame(
        data,
        columns=["monthly_orders", "avg_order_value", "avg_delivery_rating"],
    )
    df["monthly_orders"] = df["monthly_orders"].clip(1, 40).round().astype(int)
    df["avg_order_value"] = df["avg_order_value"].clip(50, 900).round(2)
    df["avg_delivery_rating"] = df["avg_delivery_rating"].clip(1.0, 5.0).round(2)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_customers()
    features = ["monthly_orders", "avg_order_value", "avg_delivery_rating"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    ks = list(range(2, 10))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    print("=" * 64)
    print("  CUSTOMER SEGMENTATION - K-MEANS + PCA")
    print("=" * 64)
    print(f"  Customers : {len(df)}")
    print("  Features  : monthly_orders, avg_order_value, avg_delivery_rating")
    print("  Scaling   : StandardScaler (all three features)")
    print("\n  Elbow Method - inertia by k")
    for k, inertia in zip(ks, inertias):
        print(f"    k={k}  inertia={inertia:,.2f}")

    # Sharp drop from k=2 to k=3, then flattening: classic elbow at k=3
    optimal_k = 3
    print(f"\n  Optimal k from elbow: {optimal_k}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, marker="o", color="#1f4e79", linewidth=2)
    ax.axvline(optimal_k, color="#c0392b", linestyle="--", label=f"Selected k = {optimal_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow Method for Food Delivery Customer Segmentation")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend()
    elbow_path = OUTPUT_DIR / "task4_elbow_method.png"
    fig.tight_layout()
    fig.savefig(elbow_path, dpi=140)
    plt.close(fig)

    model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = model.fit_predict(X_scaled)

    print("\n  Cluster sizes")
    counts = df["cluster"].value_counts().sort_index()
    for cluster_id, count in counts.items():
        print(f"    Cluster {cluster_id}: {count} customers")

    print("\n  Cluster centroids (original feature scale)")
    centres = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=features,
    )
    centres.index = [f"Cluster {i}" for i in range(optimal_k)]
    print(centres.round(2).to_string())

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for cluster_id in range(optimal_k):
        mask = df["cluster"] == cluster_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colours[cluster_id % len(colours)],
            label=f"Cluster {cluster_id}",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            s=45,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means Customer Clusters in PCA Space")
    ax.legend(title="Cluster")
    ax.grid(True, alpha=0.25)
    pca_path = OUTPUT_DIR / "task4_pca_clusters.png"
    fig.tight_layout()
    fig.savefig(pca_path, dpi=140)
    plt.close(fig)

    print(f"\n  PCA variance explained: PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%")
    print(f"  Saved elbow plot : {elbow_path}")
    print(f"  Saved PCA plot   : {pca_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
