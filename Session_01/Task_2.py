"""
Session 01 - Task 2
Comparison Table: Supervised vs Unsupervised Learning
Using real-world app examples (Instagram, Spotify, etc.)
"""

import pandas as pd

def get_comparison_table():
    comparison_data = {
        "Aspect / Feature": [
            "1. Primary Goal & Learning Type",
            "2. Input Data Requirement",
            "3. Key Real-World App Example",
            "4. Core Algorithms",
            "5. Model Evaluation Method",
            "6. Feedback & Optimization"
        ],
        "Supervised Learning": [
            "Predict target labels/outcomes for unseen data",
            "Labeled data (Inputs X paired with target labels Y)",
            "Instagram Feed Ranking (Predicting post engagement / likelihood to like based on user history)",
            "Linear Regression, Logistic Regression, Decision Trees, Random Forest, Neural Networks",
            "Accuracy, Precision, Recall, F1-Score, Mean Squared Error (MSE)",
            "Direct error measurement against ground-truth labels during training"
        ],
        "Unsupervised Learning": [
            "Discover hidden patterns, structures, and groupings without pre-existing labels",
            "Unlabeled data (Only inputs X without explicit targets Y)",
            "Spotify Daily Mix & Playlist Grouping (Clustering tracks by audio features like tempo & acousticness)",
            "K-Means, DBSCAN, Hierarchical Clustering, PCA, t-SNE",
            "Silhouette Score, Within-Cluster Sum of Squares (Inertia), Davies-Bouldin Index",
            "Heuristic structure evaluation; no explicit ground-truth target available"
        ]
    }
    return pd.DataFrame(comparison_data)

if __name__ == "__main__":
    df_comparison = get_comparison_table()
    print("=" * 100)
    print("SESSION 01 - TASK 2: Supervised vs Unsupervised Learning Comparison Table")
    print("=" * 100)
    print(df_comparison.to_string(index=False))
    print("=" * 100)
