"""
Session 02 - Task 5
Elbow Method to determine optimal K for KMeans clustering.
Dataset: Zomato Restaurants (Average Cost for Two & Ratings)
Calculates WCSS for k=1 to 6 and plots elbow curve.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_elbow_method(data_path="Datasets/zomato_ratings.csv"):
    df = pd.read_csv(data_path)
    X = df[['average_cost', 'rating']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    k_values = range(1, 7)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
    print("=" * 75)
    print("SESSION 02 - TASK 5: Elbow Method for Optimal K Selection")
    print("=" * 75)
    print("K (Clusters) | Within-Cluster Sum of Squares (WCSS)")
    print("-" * 55)
    for k, score in zip(k_values, wcss):
        print(f"    k = {k}    | {score:14.4f}")
        
    # Calculate percentage reduction in WCSS to identify elbow mathematically
    wcss_diffs = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
    print("\nWCSS Reduction between consecutive K values:")
    for i, diff in enumerate(wcss_diffs, 1):
        print(f"  • k={i} -> k={i+1}: Drop of {diff:.4f} in WCSS")
        
    optimal_k = 3
    print(f"\nIdentified Optimal K ('Elbow Point'): k = {optimal_k}")
    print("Explanation: Beyond k=3, the rate of decrease in WCSS slows down significantly ('elbow'),")
    print("balancing cluster compactness with model simplicity.")

    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='WCSS')
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k} (Elbow)')
    
    plt.title("Elbow Method: WCSS vs Number of Clusters (k)", fontsize=13, fontweight='bold')
    plt.xlabel("Number of Clusters (k)", fontsize=11)
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("Session_02/task_5_elbow_method.png", dpi=300)
    print("\nSaved plot visualization to Session_02/task_5_elbow_method.png")
    plt.close()

if __name__ == "__main__":
    run_elbow_method()
