"""
Session 03 - Task 2
Visualization of DBSCAN Clustering on Mall Customers dataset.
Plots each cluster in distinct colors and noise points in black.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def visualize_dbscan():
    df = pd.read_csv("Datasets/mall_customers.csv")
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    eps_val = 0.35
    min_samples_val = 5
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    labels = dbscan.fit_predict(X_scaled)
    
    unique_labels = sorted(list(set(labels)))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(9, 6))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points marked in black
            col = [0, 0, 0, 1]
            label_name = 'Noise (Outliers)'
            marker = 'x'
            markersize = 9
        else:
            label_name = f'Cluster {k}'
            marker = 'o'
            markersize = 7
            
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        
        if k == -1:
            plt.scatter(xy[:, 0], xy[:, 1], color=tuple(col), marker=marker, s=markersize*10,
                        label=label_name, alpha=0.85)
        else:
            plt.scatter(xy[:, 0], xy[:, 1], color=tuple(col), marker=marker, s=markersize*10,
                        label=label_name, alpha=0.85, edgecolors='black')
        
    plt.title(f"DBSCAN Clustering (eps={eps_val}, min_samples={min_samples_val})", fontsize=13, fontweight='bold')
    plt.xlabel("Annual Income (k$)", fontsize=11)
    plt.ylabel("Spending Score (1-100)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig("Session_03/task_2_dbscan_visualization.png", dpi=300)
    print("=" * 75)
    print("SESSION 03 - TASK 2: Saved plot to Session_03/task_2_dbscan_visualization.png")
    print("=" * 75)
    plt.close()

if __name__ == "__main__":
    visualize_dbscan()
