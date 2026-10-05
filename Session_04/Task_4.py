"""
Session 04 - Task 4
K-Means Clustering (k=3) on PCA-reduced Iris dataset (2 components).
Plots clusters and centroids on 2D PCA projection plane.
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_kmeans_pca_iris():
    iris = load_iris()
    X = iris.data
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA reduction to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans with k=3 on PCA transformed features
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    centroids = kmeans.cluster_centers_
    
    print("=" * 75)
    print("SESSION 04 - TASK 4: K-Means Clustering on 2D PCA Iris Data")
    print("=" * 75)
    print(f"Data Points Shape    : {X_pca.shape}")
    print(f"K-Means Cluster Count: 3")
    print("Learned Centroids in PCA Space (PC1, PC2):")
    for i, c in enumerate(centroids):
        print(f"  • Centroid {i+1}: PC1 = {c[0]:7.4f}, PC2 = {c[1]:7.4f}")
    print("-" * 75)

    # Plotting
    plt.figure(figsize=(9, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(3):
        plt.scatter(
            X_pca[cluster_labels == i, 0],
            X_pca[cluster_labels == i, 1],
            color=colors[i],
            s=55,
            alpha=0.8,
            edgecolors='black',
            label=f'KMeans Cluster {i+1}'
        )
        
    # Plot Centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color='red',
        marker='X',
        s=250,
        linewidths=2,
        edgecolors='black',
        label='Cluster Centroids',
        zorder=5
    )
    
    plt.title("K-Means (k=3) Clusters & Centroids on Iris 2D PCA Projection", fontsize=12, fontweight='bold')
    plt.xlabel("Principal Component 1 (PC1)", fontsize=11)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig("Session_04/task_4_kmeans_on_pca_iris.png", dpi=300)
    print("Saved plot visualization to Session_04/task_4_kmeans_on_pca_iris.png")
    plt.close()

if __name__ == "__main__":
    run_kmeans_pca_iris()
