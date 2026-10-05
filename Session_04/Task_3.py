"""
Session 04 - Task 3
PCA Dimension Reduction on Zomato Ratings dataset (10 features -> 3 components).
Visualizes cumulative explained variance as a line plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_zomato_pca(file_path="Datasets/zomato_ratings.csv"):
    df = pd.read_csv(file_path)
    feature_cols = df.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 1. Fit PCA with all 10 components for cumulative variance analysis
    pca_full = PCA(n_components=10)
    pca_full.fit(X_scaled)
    
    cum_var_ratio = np.cumsum(pca_full.explained_variance_ratio_)
    
    # 2. Fit PCA with top 3 components as requested
    pca_3 = PCA(n_components=3)
    X_pca_3 = pca_3.fit_transform(X_scaled)
    var_ratio_3 = pca_3.explained_variance_ratio_
    
    print("=" * 85)
    print("SESSION 04 - TASK 3: Zomato Ratings Dataset PCA (10 Features -> 3 Components)")
    print("=" * 85)
    print(f"Original Feature List ({len(feature_cols)} features):")
    print(", ".join(feature_cols))
    print("-" * 85)
    print("Individual & Cumulative Variance Ratios for Top 3 Components:")
    for idx, (var, cum) in enumerate(zip(var_ratio_3, cum_var_ratio[:3]), 1):
        print(f"  • PC{idx}: Individual Variance = {var:.4f} ({var*100:.2f}%) | Cumulative = {cum:.4f} ({cum*100:.2f}%)")
    print("-" * 85)

    # Line Plot for Cumulative Explained Variance
    plt.figure(figsize=(9, 5.5))
    components_range = range(1, 11)
    plt.plot(components_range, cum_var_ratio * 100, marker='o', color='#1f77b4', linewidth=2.5, markersize=7, label='Cumulative Explained Variance')
    
    # Highlight 3 Components mark
    plt.scatter([3], [cum_var_ratio[2] * 100], color='red', s=120, zorder=5, label=f'3 Components ({cum_var_ratio[2]*100:.1f}%)')
    plt.axvline(x=3, color='red', linestyle='--', alpha=0.7)
    
    plt.title("Zomato Ratings: Cumulative Explained Variance vs Number of Principal Components", fontsize=12, fontweight='bold')
    plt.xlabel("Number of Principal Components", fontsize=11)
    plt.ylabel("Cumulative Explained Variance (%)", fontsize=11)
    plt.xticks(components_range)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig("Session_04/task_3_zomato_pca_variance.png", dpi=300)
    print("Saved plot visualization to Session_04/task_3_zomato_pca_variance.png")
    plt.close()

if __name__ == "__main__":
    run_zomato_pca()
