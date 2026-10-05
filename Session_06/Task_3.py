"""
Session 06 - Task 3
Side-by-Side Dendrogram Comparison: Ward vs Single vs Complete Linkage on Iris Data.
Plots 3 dendrograms in a 1x3 grid and highlights structural differences.
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def compare_iris_dendrograms():
    iris = load_iris()
    X = iris.data
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    linkage_methods = ['ward', 'single', 'complete']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    print("=" * 85)
    print("SESSION 06 - TASK 3: Iris Dendrogram Linkage Comparison (Ward, Single, Complete)")
    print("=" * 85)
    
    for idx, method in enumerate(linkage_methods):
        Z = linkage(X_scaled, method=method)
        ax = axes[idx]
        
        dendrogram(Z, leaf_rotation=90, leaf_font_size=5, ax=ax)
        ax.set_title(f"Method: '{method.capitalize()}' Linkage", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Index", fontsize=9)
        ax.set_ylabel("Distance (Merge Height)", fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        print(f"• Method '{method.capitalize():<8}': Max merge height = {Z[-1, 2]:.4f}")

    plt.tight_layout()
    plt.savefig("Session_06/task_3_linkage_dendrograms_comparison.png", dpi=300)
    print("\nSaved 3-panel side-by-side dendrogram plot to Session_06/task_3_linkage_dendrograms_comparison.png")
    plt.close()

    print("\n" + "=" * 85)
    print("COMPARATIVE STRUCTURAL FINDINGS:")
    print("=" * 85)
    print("""
1. Ward Linkage:
   - Minimizes within-cluster variance, producing highly balanced, spherical, and distinct clusters.
   - Distinct tree split between Setosa (far left) and Versicolor/Virginica (right).

2. Single Linkage:
   - Uses minimum distance between points, causing severe 'chaining' where points merge sequentially one-by-one.
   - Results in a shallow tree height (max height ~1.76) with poorly defined cluster boundaries.

3. Complete Linkage:
   - Uses maximum distance between points, enforcing compact clusters with high top-level merge heights (max height ~7.12).
   - Produces balanced clusters similar to Ward, but is more sensitive to outliers.
""")

if __name__ == "__main__":
    compare_iris_dendrograms()
