"""
Session 04 - Task 2
2D Projection Visualization of PCA-transformed Iris dataset.
Colors points by species class (Setosa, Versicolor, Virginica), akin to Spotify genre clustering.
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def visualize_iris_pca():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    var_ratio = pca.explained_variance_ratio_
    
    plt.figure(figsize=(9, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    for i, (target, name) in enumerate(zip(range(3), target_names)):
        plt.scatter(
            X_pca[y == target, 0],
            X_pca[y == target, 1],
            color=colors[i],
            marker=markers[i],
            s=60,
            alpha=0.85,
            edgecolors='black',
            label=f'Iris {name.capitalize()}'
        )
        
    plt.title("2D PCA Projection of Iris Species Clusters", fontsize=13, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig("Session_04/task_2_iris_pca_2d.png", dpi=300)
    print("=" * 75)
    print("SESSION 04 - TASK 2: Saved plot to Session_04/task_2_iris_pca_2d.png")
    print("=" * 75)
    plt.close()

if __name__ == "__main__":
    visualize_iris_pca()
