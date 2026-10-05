"""
Section D — STEP 2/3: Corrected K-Means + Elbow + PCA program

Fixes applied without relying on AI for the debug pass:
1. StandardScaler before K-Means and PCA
2. random_state=42 and n_init=10 for reproducible clustering
3. Elbow loop k=2 to 9 (avoid k=1)
4. PCA axes labelled PC1 and PC2, plus a colour legend
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def make_customers(n: int = 180, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
    data = np.vstack(
        [
            np.column_stack(
                [
                    rng.integers(1, 5, size=n1),
                    rng.normal(180, 30, size=n1),
                    rng.normal(3.55, 0.28, size=n1),
                ]
            ),
            np.column_stack(
                [
                    rng.integers(6, 14, size=n2),
                    rng.normal(320, 35, size=n2),
                    rng.normal(4.2, 0.22, size=n2),
                ]
            ),
            np.column_stack(
                [
                    rng.integers(15, 28, size=n3),
                    rng.normal(610, 50, size=n3),
                    rng.normal(4.7, 0.16, size=n3),
                ]
            ),
        ]
    )
    rng.shuffle(data)
    df = pd.DataFrame(data, columns=["monthly_orders", "avg_spend", "avg_rating"])
    df["monthly_orders"] = df["monthly_orders"].clip(1, 40).round().astype(int)
    df["avg_spend"] = df["avg_spend"].clip(50, 900).round(2)
    df["avg_rating"] = df["avg_rating"].clip(1.0, 5.0).round(2)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_customers()
    features = ["monthly_orders", "avg_spend", "avg_rating"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    ks = list(range(2, 10))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    print("Elbow Method inertias (scaled features)")
    for k, inertia in zip(ks, inertias):
        print(f"  k={k}: {inertia:,.2f}")

    optimal_k = 3
    print(f"\nOptimal k selected from the elbow: {optimal_k}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, marker="o", color="#1f4e79")
    ax.axvline(optimal_k, color="#c0392b", linestyle="--", label=f"k = {optimal_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — Food Delivery Customers")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_d_elbow.png", dpi=140)
    plt.close(fig)

    model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    df["cluster"] = labels

    centres = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=features,
    ).round(2)

    print("\nCluster sizes and centroids (original units)")
    for cluster_id in range(optimal_k):
        count = int((labels == cluster_id).sum())
        row = centres.iloc[cluster_id]
        print(f"  Cluster {cluster_id}: {count} customers")
        print(
            f"    monthly_orders={row['monthly_orders']:.2f}, "
            f"avg_spend={row['avg_spend']:.2f}, "
            f"avg_rating={row['avg_rating']:.2f}"
        )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_id in range(optimal_k):
        mask = labels == cluster_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colours[cluster_id],
            label=f"Cluster {cluster_id}",
            alpha=0.85,
            s=45,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Corrected K-Means clusters in PCA space")
    ax.legend(title="Cluster")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_d_pca_clusters.png", dpi=140)
    plt.close(fig)
    print(f"\nPlots saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
