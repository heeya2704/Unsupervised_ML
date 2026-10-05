"""
Session 07 - Task 3
Scatterplot of Annual Income vs Spending Score using Seaborn/Matplotlib.
Visualizes overall customer distribution pattern.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_income_vs_spending():
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
    
    print("=" * 85)
    print("SESSION 07 - TASK 3: Scatterplot of Annual Income vs Spending Score")
    print("=" * 85)
    
    # Configure plot aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    
    # Seaborn scatter plot
    scatter = sns.scatterplot(
        data=df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Gender',
        palette={'Male': '#1f77b4', 'Female': '#e377c2'},
        s=80,
        alpha=0.85,
        edgecolor='k',
        linewidth=0.5
    )
    
    plt.title("Mall Customer Distribution: Annual Income vs Spending Score", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Annual Income (k$)", fontsize=11, fontweight='bold')
    plt.ylabel("Spending Score (1-100)", fontsize=11, fontweight='bold')
    plt.legend(title="Gender", loc="upper right")
    plt.tight_layout()
    
    # Save image artifact
    output_dir = "Session_07"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "task_3_customer_scatterplot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Successfully generated and saved plot to: {output_path}")
    plt.close()
    print("=" * 85)

if __name__ == "__main__":
    plot_income_vs_spending()
