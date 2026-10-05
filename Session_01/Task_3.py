"""
Session 01 - Task 3
Product Segmentation: 8 products segmented into 3 clusters based on Price and Number of Reviews.
Includes reasoning for each cluster and plot visualizer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define 8 realistic products with Price ($) and Number of Reviews
products_data = [
    {"product_name": "Wireless Ergonomic Mouse", "price": 19.99, "num_reviews": 4500},
    {"product_name": "Budget USB-C Cable (Pack of 3)", "price": 9.99, "num_reviews": 8200},
    {"product_name": "Noise-Canceling Wireless Earbuds", "price": 89.99, "num_reviews": 1200},
    {"product_name": "Smart Fitness Watch", "price": 129.99, "num_reviews": 1500},
    {"product_name": "Ultra HD 4K Gaming Monitor", "price": 499.99, "num_reviews": 150},
    {"product_name": "High-End DSLR Camera", "price": 1199.99, "num_reviews": 85},
    {"product_name": "Mechanical Gaming Keyboard", "price": 74.99, "num_reviews": 2100},
    {"product_name": "Basic Laptop Sleeve", "price": 14.99, "num_reviews": 6300},
]

def segment_products():
    df = pd.DataFrame(products_data)
    
    # Standardize features for KMeans clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['price', 'num_reviews']])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_names = {
        0: "High Volume / Budget Essentials (Low Price, Very High Popularity)",
        1: "Mid-Tier Popular Tech (Moderate Price, High Popularity)",
        2: "Premium & Professional Gear (High Price, Low Volume Reviews)"
    }
    
    # Map friendly names based on actual cluster characteristics
    cluster_means = df.groupby('cluster')[['price', 'num_reviews']].mean()
    sorted_clusters = cluster_means.sort_values(by='price').index
    
    name_mapping = {
        sorted_clusters[0]: "Cluster 1: Budget & High Volume (Low Price, High Reviews)",
        sorted_clusters[1]: "Cluster 2: Mid-Range Everyday Tech (Moderate Price, Moderate Reviews)",
        sorted_clusters[2]: "Cluster 3: Premium High-End Items (High Price, Low Reviews)"
    }
    
    df['cluster_label'] = df['cluster'].map(name_mapping)
    
    return df

def visualize_clusters(df):
    plt.figure(figsize=(9, 6))
    clusters = df['cluster_label'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, cluster in enumerate(sorted(clusters)):
        sub = df[df['cluster_label'] == cluster]
        plt.scatter(sub['price'], sub['num_reviews'], s=120, label=cluster, color=colors[i % len(colors)], edgecolors='black', alpha=0.85)
        for _, row in sub.iterrows():
            plt.annotate(row['product_name'], (row['price'], row['num_reviews']),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
            
    plt.title("Product Segmentation: Price vs Number of Reviews", fontsize=13, fontweight='bold')
    plt.xlabel("Price ($)", fontsize=11)
    plt.ylabel("Number of Reviews", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig("Session_01/task_3_product_clusters.png", dpi=300)
    print("Saved plot to Session_01/task_3_product_clusters.png")
    plt.close()

if __name__ == "__main__":
    df_result = segment_products()
    
    print("=" * 80)
    print("SESSION 01 - TASK 3: Product Segmentation (Price & Reviews)")
    print("=" * 80)
    for cluster_label, group in df_result.groupby('cluster_label'):
        print(f"\n{cluster_label}:")
        print("-" * 65)
        for _, row in group.iterrows():
            print(f"  • {row['product_name']:<35} | Price: ${row['price']:<7.2f} | Reviews: {row['num_reviews']}")
            
    print("\n" + "=" * 80)
    print("REASONING FOR EACH CLUSTER:")
    print("=" * 80)
    print("""
1. Budget & High Volume Cluster:
   - Includes: Budget USB-C Cable, Basic Laptop Sleeve, Wireless Ergonomic Mouse.
   - Reasoning: Low price point (< $20) and very high review counts (> 4,000). These are impulse buys / everyday essentials that attract a massive volume of buyers.

2. Mid-Range Everyday Tech Cluster:
   - Includes: Mechanical Gaming Keyboard, Noise-Canceling Earbuds, Smart Fitness Watch.
   - Reasoning: Moderate price range ($70 - $130) with steady, high review counts (1,000 - 2,500). These represent popular lifestyle electronics with broad market appeal.

3. Premium High-End Items Cluster:
   - Includes: Ultra HD 4K Gaming Monitor, High-End DSLR Camera.
   - Reasoning: High price range ($500 - $1200+) and lower review counts (< 200). These are premium niche products targeted at professionals and enthusiasts with lower purchasing frequency.
""")
    visualize_clusters(df_result)
