"""
Session 03 - Task 5
DBSCAN Clustering on Non-Spherical Dataset (make_moons with noise).
Identifies and prints the indices of all points labeled as noise (outliers).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

def dbscan_non_spherical():
    # Generate non-spherical moon-shaped dataset with added noise points
    X, y_true = make_moons(n_samples=300, noise=0.12, random_state=42)
    
    # Add 10 explicit random outliers
    np.random.seed(101)
    outliers = np.random.uniform(low=-1.8, high=2.5, size=(10, 2))
    X_combined = np.vstack([X, outliers])
    
    # Fit DBSCAN
    eps_val = 0.18
    min_samples_val = 5
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    labels = dbscan.fit_predict(X_combined)
    
    # Identify indices of points labeled as noise (-1)
    noise_indices = np.where(labels == -1)[0]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print("=" * 85)
    print("SESSION 03 - TASK 5: DBSCAN on Non-Spherical Moons Dataset")
    print("=" * 85)
    print(f"Total Dataset Size   : {len(X_combined)} points ({len(X)} moon samples + 10 random outliers)")
    print(f"DBSCAN Parameters    : eps = {eps_val}, min_samples = {min_samples_val}")
    print(f"Clusters Detected    : {n_clusters} (Moons)")
    print(f"Noise Points Count   : {len(noise_indices)}")
    print("-" * 85)
    print("Indices of Noise / Outlier Points:")
    print(noise_indices.tolist())
    print("-" * 85)

    # Plot results
    plt.figure(figsize=(9, 6))
    
    unique_labels = sorted(list(set(labels)))
    colors = [plt.cm.Set1(each) for each in np.linspace(0, 1, max(len(unique_labels), 1))]
    
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = X_combined[class_member_mask]
        
        if k == -1:
            plt.scatter(xy[:, 0], xy[:, 1], color='black', marker='x', s=70, label=f'Noise (Outliers) [n={len(xy)}]', zorder=4)
        else:
            plt.scatter(xy[:, 0], xy[:, 1], color=colors[k], marker='o', s=40, label=f'Cluster {k+1} (Moon)', alpha=0.85, edgecolors='k')
            
    plt.title("DBSCAN Clustering on Non-Spherical Moons Dataset", fontsize=13, fontweight='bold')
    plt.xlabel("Feature 1 (X1)", fontsize=11)
    plt.ylabel("Feature 2 (X2)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig("Session_03/task_5_non_spherical_dbscan.png", dpi=300)
    print("Saved plot visualization to Session_03/task_5_non_spherical_dbscan.png")
    plt.close()

if __name__ == "__main__":
    dbscan_non_spherical()
