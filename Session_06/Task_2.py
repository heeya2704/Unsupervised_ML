"""
Session 06 - Task 2
Cutting Ward Dendrogram at Height to form 3 Distinct Clusters.
Uses scipy.cluster.hierarchy.fcluster to assign clusters and print sample counts.
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd

def cut_dendrogram():
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    target_names = iris.target_names
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute Ward linkage
    Z = linkage(X_scaled, method='ward')
    
    # Cut dendrogram at height h=10.0 (or maxclust=3)
    cut_height = 10.0
    cluster_labels = fcluster(Z, t=cut_height, criterion='distance')
    
    print("=" * 75)
    print("SESSION 06 - TASK 2: Dendrogram Cut & Cluster Assignment (3 Clusters)")
    print("=" * 75)
    print(f"Chosen Cut Height (Distance Threshold) : h = {cut_height}")
    print(f"Total Data Samples Processed           : {len(cluster_labels)}")
    print("-" * 75)
    
    # Sample counts per assigned cluster
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("Number of Samples in Each Assigned Cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"  • Cluster {cluster_id}: {count} samples")
        
    print("-" * 75)
    print("Cross-Tabulation with Actual Ground-Truth Iris Species:")
    df_compare = pd.DataFrame({'Assigned_Cluster': cluster_labels, 'Actual_Species': [target_names[i] for i in y_true]})
    ct = pd.crosstab(df_compare['Assigned_Cluster'], df_compare['Actual_Species'])
    print(ct.to_string())
    print("=" * 75)

if __name__ == "__main__":
    cut_dendrogram()
