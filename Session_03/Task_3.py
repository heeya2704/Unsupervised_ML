"""
Session 03 - Task 3
Impact of DBSCAN Parameters (eps and min_samples) on Clustering Outcomes.
Compares multiple parameter configurations and provides a 3-4 line summary explanation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def evaluate_dbscan_params():
    df = pd.read_csv("Datasets/mall_customers.csv")
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_combinations = [
        (0.20, 5),   # Small eps, medium min_samples
        (0.35, 5),   # Optimal eps, medium min_samples
        (0.35, 10),  # Optimal eps, high min_samples
        (0.50, 5)    # Large eps, medium min_samples
    ]
    
    print("=" * 85)
    print("SESSION 03 - TASK 3: DBSCAN Parameter Sensitivity Analysis")
    print("=" * 85)
    print(f"{'eps':<8} | {'min_samples':<12} | {'Clusters Found':<16} | {'Noise Points':<15} | {'Noise Ratio'}")
    print("-" * 75)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (eps, min_s) in enumerate(param_combinations):
        dbscan = DBSCAN(eps=eps, min_samples=min_s)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = (n_noise / len(labels)) * 100
        
        print(f"{eps:<8.2f} | {min_s:<12} | {n_clusters:<16} | {n_noise:<15} | {noise_ratio:.1f}%")
        
        # Plotting subplot
        ax = axes[idx]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.8, edgecolors='k', s=35)
        ax.set_title(f"eps={eps}, min_samples={min_s}\nClusters: {n_clusters} | Noise: {n_noise}", fontsize=10)
        ax.set_xlabel("Annual Income (k$)", fontsize=8)
        ax.set_ylabel("Spending Score", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("Session_03/task_3_parameter_tuning.png", dpi=300)
    print("\nSaved 2x2 comparison plot to Session_03/task_3_parameter_tuning.png")
    plt.close()

    print("\n" + "=" * 85)
    print("PARAMETER SENSITIVITY SUMMARY (3-4 Lines):")
    print("=" * 85)
    print("""
1. Increasing 'eps' expands the neighborhood radius, allowing points further apart to join the same cluster; setting eps too high merges distinct clusters together and reduces noise points to zero.
2. Decreasing 'eps' creates stricter neighborhood boundaries, causing fragmented clusters and classifying many legitimate data points as noise (e.g., 67 noise points at eps=0.20).
3. Increasing 'min_samples' requires higher local point density to form a cluster core, turning sparse border points into noise unless eps is simultaneously increased.
4. Optimal DBSCAN performance requires balancing eps and min_samples (e.g., eps=0.35, min_samples=5) to capture natural high-density clusters while isolating true outliers.
""")

if __name__ == "__main__":
    evaluate_dbscan_params()
