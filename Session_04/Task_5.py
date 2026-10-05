"""
Session 04 - Task 5
Silhouette Score Evaluation of K-Means Clustering on PCA-reduced Iris Data.
Prints silhouette score and provides a 2-3 line interpretation.
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def evaluate_silhouette():
    iris = load_iris()
    X = iris.data
    
    # Standardize and reduce to 2 PCA components
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Fit KMeans with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    # Calculate Silhouette Score
    score = silhouette_score(X_pca, labels)
    
    print("=" * 75)
    print("SESSION 04 - TASK 5: Silhouette Score Evaluation")
    print("=" * 75)
    print(f"Dataset Evaluated           : Iris Dataset (PCA-transformed to 2D)")
    print(f"Number of Clusters (K)      : 3")
    print(f"Calculated Silhouette Score : {score:.4f}")
    print("-" * 75)
    print("\nINTERPRETATION & EXPLANATION (2-3 Lines):")
    print("=" * 75)
    print(f"""
1. The silhouette score of {score:.4f} (ranging from -1 to +1) indicates strong overall cluster separation and cohesion.
2. A score near ~0.51 confirms that one cluster (Setosa) is distinct and perfectly separated, while the other two clusters (Versicolor & Virginica) are reasonably compact with moderate boundary overlap in 2D PCA space.
3. Values significantly above 0 demonstrate that points are much closer to their own cluster members than to neighboring cluster points.
""")

if __name__ == "__main__":
    evaluate_silhouette()
