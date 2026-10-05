"""
Session 06 - Task 1
Hierarchical Clustering on Iris dataset using Ward's linkage.
Plots resulting dendrogram using scipy.cluster.hierarchy.dendrogram.
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def plot_iris_ward_dendrogram():
    iris = load_iris()
    X = iris.data
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform Ward linkage hierarchical clustering
    Z = linkage(X_scaled, method='ward')
    
    print("=" * 75)
    print("SESSION 06 - TASK 1: Hierarchical Clustering (Ward Linkage) on Iris")
    print("=" * 75)
    print(f"Dataset Shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Top 3 Merges in Linkage Matrix Z (Height / Distance):")
    print(Z[-3:])
    print("-" * 75)

    # Plot Dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=6, color_threshold=10)
    
    plt.title("Iris Dataset Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight='bold')
    plt.xlabel("Sample Index", fontsize=11)
    plt.ylabel("Euclidean Distance (Ward Linkage Height)", fontsize=11)
    plt.axhline(y=10, color='red', linestyle='--', label='Cut Height (h=10 for 3 Clusters)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("Session_06/task_1_iris_ward_dendrogram.png", dpi=300)
    print("Saved plot visualization to Session_06/task_1_iris_ward_dendrogram.png")
    plt.close()

if __name__ == "__main__":
    plot_iris_ward_dendrogram()
