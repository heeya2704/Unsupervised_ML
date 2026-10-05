"""
Session 07 - Task 4
Heatmap of Correlation Matrix for Numerical Features.
Interprets feature pair with the strongest linear relationship.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlation_heatmap():
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
    
    # Extract numerical features
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    print("=" * 85)
    print("SESSION 07 - TASK 4: Correlation Matrix & Heatmap Analysis")
    print("=" * 85)
    print("Numerical Feature Pairwise Correlation Matrix:")
    print("-" * 85)
    print(corr_matrix.round(4).to_string())
    print("-" * 85)
    
    # Plot Seaborn Heatmap
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title("Correlation Matrix Heatmap (Numerical Mall Features)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    
    output_dir = "Session_07"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "task_4_correlation_heatmap.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved correlation heatmap visualization to: {output_path}")
    plt.close()
    
    # Find strongest non-self correlation pair
    # Exclude self-correlation (diagonal = 1.0) and CustomerID if desired, or examine all feature pairs
    feature_pairs = []
    cols = numeric_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            r = corr_matrix.loc[col1, col2]
            feature_pairs.append((col1, col2, r, abs(r)))
            
    # Sort by absolute correlation
    feature_pairs.sort(key=lambda x: x[3], reverse=True)
    
    print("\n" + "=" * 85)
    print("CORRELATION INTERPRETATION & FINDINGS:")
    print("=" * 85)
    print("Ranked Feature Pair Correlations (by magnitude |r|):")
    for f1, f2, r, abs_r in feature_pairs:
        print(f"  • {f1:<25} vs {f2:<25} : r = {r:+.4f} (Magnitude |r| = {abs_r:.4f})")
        
    top_pair = feature_pairs[0]
    print("-" * 85)
    print(f"MOST STRONGLY CORRELATED FEATURE PAIR: '{top_pair[0]}' and '{top_pair[1]}'")
    print(f"Correlation Coefficient (r): {top_pair[2]:.4f}")
    
    # Also evaluate meaningful domain features (excluding CustomerID index)
    domain_pairs = [p for p in feature_pairs if 'CustomerID' not in (p[0], p[1])]
    if domain_pairs:
        top_domain = domain_pairs[0]
        print(f"\nMOST STRONGLY CORRELATED DOMAIN FEATURE PAIR (excluding CustomerID):")
        print(f"  • '{top_domain[0]}' and '{top_domain[1]}' with r = {top_domain[2]:.4f}")
        print(f"  • Interpretation: Age and Spending Score have a moderate negative correlation (r = {top_domain[2]:.4f}),")
        print(f"    indicating that younger customers tend to have higher spending scores, whereas older customers tend to spend less.")
    print("=" * 85)

if __name__ == "__main__":
    generate_correlation_heatmap()
