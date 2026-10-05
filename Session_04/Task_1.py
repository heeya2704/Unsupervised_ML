"""
Session 04 - Task 1
PCA Dimension Reduction on Iris Dataset (4 features to 2 Principal Components).
Prints explained variance ratio for PC1 and PC2.
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_pca_iris():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    # Standardize features (Mean=0, Variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = var_ratio.sum()
    
    print("=" * 75)
    print("SESSION 04 - TASK 1: PCA Dimension Reduction on Iris Dataset")
    print("=" * 75)
    print(f"Original Feature Count : {X.shape[1]} ({', '.join(feature_names)})")
    print(f"Reduced Feature Count  : {X_pca.shape[1]} (PC1, PC2)")
    print("-" * 75)
    print(f"Explained Variance Ratio (PC1): {var_ratio[0]:.4f} ({var_ratio[0]*100:.2f}%)")
    print(f"Explained Variance Ratio (PC2): {var_ratio[1]:.4f} ({var_ratio[1]*100:.2f}%)")
    print(f"Cumulative Explained Variance : {cum_var_ratio:.4f} ({cum_var_ratio*100:.2f}%)")
    print("-" * 75)
    print("Sample PCA Transformed Data (First 5 Rows):")
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Species'] = [iris.target_names[i] for i in y]
    print(df_pca.head())

if __name__ == "__main__":
    run_pca_iris()
