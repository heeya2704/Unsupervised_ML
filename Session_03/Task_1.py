"""
Session 03 - Task 1
DBSCAN Clustering on Mall Customers dataset (Annual Income vs Spending Score)
Prints number of clusters found and noise points detected.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def run_dbscan_mall_customers(file_path="Datasets/mall_customers.csv"):
    df = pd.read_csv(file_path)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply DBSCAN
    eps_val = 0.35
    min_samples_val = 5
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    labels = dbscan.fit_predict(X_scaled)
    
    # Count clusters and noise points (-1 label)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print("=" * 75)
    print("SESSION 03 - TASK 1: DBSCAN Clustering on Mall Customers Dataset")
    print("=" * 75)
    print(f"Dataset Loaded : {file_path} (Shape: {df.shape})")
    print(f"Features Used  : Annual Income (k$) & Spending Score (1-100)")
    print(f"DBSCAN Hyperparameters: eps = {eps_val}, min_samples = {min_samples_val}")
    print("-" * 75)
    print(f"Number of Clusters Found : {n_clusters}")
    print(f"Number of Noise Points   : {n_noise} ({n_noise/len(labels)*100:.1f}% of total data)")
    print("-" * 75)
    
    cluster_counts = pd.Series(labels).value_counts()
    for lbl, count in cluster_counts.items():
        name = "Noise (-1)" if lbl == -1 else f"Cluster {lbl}"
        print(f"  • {name:<12}: {count} points")

    return df, X_scaled, labels, n_clusters, n_noise

if __name__ == "__main__":
    run_dbscan_mall_customers()
