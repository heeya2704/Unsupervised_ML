"""
Session 07 - Task 5
K-Means Clustering (k=3) on Annual Income and Spending Score.
Adds 'Cluster' column to DataFrame and plots cluster segments with centroids.
Constraint: Uses only Annual Income and Spending Score, random_state=42.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def run_kmeans_customer_segmentation():
    possible_paths = [
        "Datasets/mall_customers.csv",
        "../Datasets/mall_customers.csv",
        "c:/Users/heeya/OneDrive/Documents/TOPS/Machine Learning/Unsupervised_ML/Assignment/Datasets/mall_customers.csv"
    ]
    
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break
            
    df = pd.read_csv(file_path)
    
    # Select target features according to constraint
    feature_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[feature_cols].values
    
    # Initialize and fit K-Means with n_clusters=3, random_state=42
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    print("=" * 85)
    print("SESSION 07 - TASK 5: K-Means Customer Segmentation (k=3, random_state=42)")
    print("=" * 85)
    print(f"Features Used for Clustering: {feature_cols}")
    print(f"Number of Clusters (k)      : 3")
    print(f"Random State Seed           : 42")
    print(f"Inertia (WCSS)              : {kmeans.inertia_:.4f}")
    print("-" * 85)
    
    print("\nLearned Cluster Centroids Coordinates:")
    for idx, center in enumerate(centroids):
        print(f"  • Cluster {idx}: Annual Income = {center[0]:.2f} k$ | Spending Score = {center[1]:.2f}")
        
    print("\n" + "-" * 85)
    print("Customer Count per Cluster:")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = (count / len(df)) * 100
        print(f"  • Cluster {cluster_id}: {count} customers ({pct:.2f}%)")
        
    print("\n" + "-" * 85)
    print("Mean Feature Values by Assigned Cluster:")
    cluster_summary = df.groupby('Cluster')[feature_cols + ['Age']].mean().round(2)
    print(cluster_summary.to_string())
    
    print("\n" + "-" * 85)
    print("First 10 DataFrame Rows with 'Cluster' Column:")
    print(df.head(10).to_string(index=False))
    
    # Plot K-Means Clusters & Centroids
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    
    palette = sns.color_palette("Set1", n_colors=3)
    
    for cluster_id in range(3):
        cluster_data = df[df['Cluster'] == cluster_id]
        plt.scatter(
            cluster_data['Annual Income (k$)'],
            cluster_data['Spending Score (1-100)'],
            s=70,
            color=palette[cluster_id],
            label=f"Cluster {cluster_id}",
            alpha=0.8,
            edgecolor='k',
            linewidth=0.5
        )
        
    # Plot Centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=250,
        color='red',
        marker='X',
        edgecolor='black',
        linewidth=2,
        label='Cluster Centroids',
        zorder=10
    )
    
    plt.title("K-Means Customer Segmentation (k=3, random_state=42)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Annual Income (k$)", fontsize=11, fontweight='bold')
    plt.ylabel("Spending Score (1-100)", fontsize=11, fontweight='bold')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    output_dir = "Session_07"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "task_5_kmeans_customer_clusters.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved segmentation plot visualization to: {output_path}")
    plt.close()
    print("=" * 85)
    return df

if __name__ == "__main__":
    run_kmeans_customer_segmentation()
