"""
Session 03 - Task 4
Comparative Analysis: DBSCAN vs K-Means on Mall Customers Dataset
Uses the same features (Annual Income vs Spending Score) and cluster count (k=5).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

def compare_dbscan_kmeans():
    df = pd.read_csv("Datasets/mall_customers.csv")
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Run DBSCAN
    eps_val, min_s = 0.35, 5
    dbscan = DBSCAN(eps=eps_val, min_samples=min_s)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Number of clusters in DBSCAN (excluding noise)
    k_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    # 2. Run KMeans matching DBSCAN cluster count (k=5)
    kmeans = KMeans(n_clusters=k_dbscan, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    print("=" * 85)
    print("SESSION 03 - TASK 4: DBSCAN vs K-Means Side-by-Side Comparison")
    print("=" * 85)
    print(f"Dataset Features Used  : Annual Income (k$) & Spending Score (1-100)")
    print(f"Number of Clusters (K) : {k_dbscan}")
    print(f"DBSCAN Noise Points    : {list(dbscan_labels).count(-1)}")
    print(f"KMeans Noise Handling  : Force-assigns ALL points to nearest centroid (0 noise points)")

    # Side-by-side Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # DBSCAN Plot
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='tab10', s=45, alpha=0.85, edgecolors='k')
    ax1.set_title(f"DBSCAN Clustering (k={k_dbscan})\nIsolates Outliers as Noise (-1)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Annual Income (k$)", fontsize=10)
    ax1.set_ylabel("Spending Score (1-100)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # KMeans Plot
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10', s=45, alpha=0.85, edgecolors='k')
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax2.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='KMeans Centroids', zorder=5)
    ax2.set_title(f"K-Means Clustering (k={k_dbscan})\nAssumes Convex / Spherical Clusters", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Annual Income (k$)", fontsize=10)
    ax2.set_ylabel("Spending Score (1-100)", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("Session_03/task_4_dbscan_vs_kmeans.png", dpi=300)
    print("\nSaved side-by-side plot to Session_03/task_4_dbscan_vs_kmeans.png")
    plt.close()

    print("\n" + "=" * 85)
    print("COMPARATIVE FINDINGS & ANALYSIS:")
    print("=" * 85)
    print("""
1. Outlier Handling:
   - DBSCAN inherently detects low-density regions and flags anomalous data points as noise (labeled -1).
   - K-Means has NO built-in outlier detection mechanism and forces noisy/extreme values into the nearest cluster centroid, potentially distorting cluster boundaries.

2. Cluster Shape Flexibility:
   - DBSCAN forms density-connected components of arbitrary non-spherical shapes (e.g., elongated, crescent, or ring shapes).
   - K-Means minimizes squared Euclidean distance, assuming spherical/convex clusters of similar density and equal variance around centroids.

3. Hyperparameter Dependency:
   - DBSCAN requires tuning 'eps' and 'min_samples' without requiring prior knowledge of 'k'.
   - K-Means requires explicitly specifying the exact number of clusters 'k' beforehand.
""")

if __name__ == "__main__":
    compare_dbscan_kmeans()
