# Unsupervised Machine Learning Assignments (Sessions 1 - 7)

This repository contains Python solutions, datasets, and visualization artifacts for **Sessions 01 to 07** of the Unsupervised Machine Learning course.

---

## 📁 Repository Structure

```
Assignment/
├── Datasets/
│   ├── mall_customers.csv          # Mall customer segmentation dataset (200 records)
│   └── zomato_ratings.csv          # Zomato restaurant ratings & features (150 records, 10 features)
├── Session_01/
│   ├── Task_1.py                   # User dataset generator (List of Dictionaries)
│   ├── Task_2.py                   # Supervised vs Unsupervised comparison table
│   ├── Task_3.py                   # Product segmentation (Price vs Reviews) & visualizer
│   ├── Task_4.py                   # Real-world scenarios (Zomato, Flipkart, Spotify)
│   ├── Task_5.py                   # Kaggle/UCI Mall Customer Dataset analysis
│   └── task_3_product_clusters.png # Generated product cluster plot
├── Session_02/
│   ├── Task_1.py                   # Food delivery 2D locations manual K-Means assignment
│   ├── Task_2.py                   # assign_clusters() with Euclidean distance metric
│   ├── Task_3.py                   # assign_clusters() supporting Euclidean & Manhattan metrics
│   ├── Task_4.py                   # update_centroids() mean vector calculation
│   ├── Task_5.py                   # Elbow Method (WCSS for k=1..6) & optimal K selection
│   ├── task_1_delivery_clusters.png# Initial delivery location clusters plot
│   └── task_5_elbow_method.png     # WCSS Elbow curve plot
├── Session_03/
│   ├── Task_1.py                   # DBSCAN on Mall Customers (Annual Income vs Spending Score)
│   ├── Task_2.py                   # DBSCAN cluster & noise visualization plot
│   ├── Task_3.py                   # Parameter tuning (eps & min_samples sensitivity analysis)
│   ├── Task_4.py                   # DBSCAN vs K-Means side-by-side comparison
│   ├── Task_5.py                   # DBSCAN on non-spherical dataset (make_moons + outliers)
│   ├── task_2_dbscan_visualization.png
│   ├── task_3_parameter_tuning.png
│   ├── task_4_dbscan_vs_kmeans.png
│   └── task_5_non_spherical_dbscan.png
├── Session_04/
│   ├── Task_1.py                   # PCA on Iris dataset (4 -> 2 components & explained variance)
│   ├── Task_2.py                   # 2D PCA projection scatter plot of Iris species
│   ├── Task_3.py                   # Zomato 10-feature PCA reduction & cumulative variance plot
│   ├── Task_4.py                   # K-Means (k=3) on 2D PCA Iris data with centroids
│   ├── Task_5.py                   # Silhouette score evaluation of K-Means on PCA Iris data
│   ├── task_2_iris_pca_2d.png
│   ├── task_3_zomato_pca_variance.png
│   └── task_4_kmeans_on_pca_iris.png
├── Session_05/
│   ├── Task_1.py                   # 6x6 Euclidean distance matrix calculation for delivery orders
│   ├── Task_2.py                   # Agglomerative hierarchical clustering & dendrogram for 8 songs
│   ├── Task_3.py                   # Side-by-side linkage comparison (Single, Complete, Average)
│   ├── Task_4.py                   # Dual distance calculator function (Euclidean & Manhattan)
│   ├── Task_5.py                   # Optimal linkage selection analysis for movie recommendations
│   ├── task_2_song_dendrogram.png
│   └── task_3_linkage_comparison.png
├── Session_06/
│   ├── Task_1.py                   # Ward linkage hierarchical clustering on Iris dataset & dendrogram
│   ├── Task_2.py                   # Dendrogram cutting at height h=10 (3 clusters) & confusion matrix
│   ├── Task_3.py                   # 3-panel dendrogram comparison on Iris (Ward vs Single vs Complete)
│   ├── Task_4.py                   # Interpretation guide for dendrogram branch heights & cluster structure
│   ├── task_1_iris_ward_dendrogram.png
│   └── task_3_linkage_dendrograms_comparison.png
├── Session_07/
│   ├── Task_1.py                   # Load Kaggle Customer Segmentation dataset & display top 10 rows
│   ├── Task_2.py                   # Check missing values in Age, Annual Income, and Spending Score
│   ├── Task_3.py                   # Scatterplot of Annual Income vs Spending Score
│   ├── Task_4.py                   # Seaborn correlation heatmap & strongest feature pair interpretation
│   ├── Task_5.py                   # K-Means (k=3, random_state=42) customer segmentation & Cluster column
│   ├── Session_07_Customer_Segmentation.ipynb # Complete interactive Jupyter Notebook
│   ├── task_3_customer_scatterplot.png
│   ├── task_4_correlation_heatmap.png
│   └── task_5_kmeans_customer_clusters.png
├── Utils/
│   └── create_datasets.py          # Synthetic dataset creation utility script
├── requirements.txt                # Required Python packages
└── README.md                       # Documentation & instructions
```

---

## ⚡ Setup & Execution

### 1. Installation
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running Individual Tasks
Execute any script directly from the root workspace directory:

```bash
# Session 1
python Session_01/Task_1.py
python Session_01/Task_2.py
python Session_01/Task_3.py
python Session_01/Task_4.py
python Session_01/Task_5.py

# Session 2
python Session_02/Task_1.py
python Session_02/Task_2.py
python Session_02/Task_3.py
python Session_02/Task_4.py
python Session_02/Task_5.py

# Session 3
python Session_03/Task_1.py
python Session_03/Task_2.py
python Session_03/Task_3.py
python Session_03/Task_4.py
python Session_03/Task_5.py

# Session 4
python Session_04/Task_1.py
python Session_04/Task_2.py
python Session_04/Task_3.py
python Session_04/Task_4.py
python Session_04/Task_5.py

# Session 5
python Session_05/Task_1.py
python Session_05/Task_2.py
python Session_05/Task_3.py
python Session_05/Task_4.py
python Session_05/Task_5.py

# Session 6
python Session_06/Task_1.py
python Session_06/Task_2.py
python Session_06/Task_3.py
python Session_06/Task_4.py

# Session 7
python Session_07/Task_1.py
python Session_07/Task_2.py
python Session_07/Task_3.py
python Session_07/Task_4.py
python Session_07/Task_5.py
```

---

## 📊 Summary of Tasks & Key Results

### Session 01: Introduction to Unsupervised Learning
- **Task 1**: Generated 10 user dictionaries containing `daily_app_opens` and `avg_session_duration_min`.
- **Task 2**: Formatted comparative analysis table between Supervised & Unsupervised Learning across 6 core criteria.
- **Task 3**: Segmented 8 products into 3 distinct clusters (*Budget & High Volume*, *Mid-Range Tech*, *Premium Gear*) based on Price and Review counts.
- **Task 4**: Detailed 3 industry use cases (Zomato restaurant hubs, Flipkart shopper personas, Spotify acoustic micro-genres).
- **Task 5**: Structured overview and summary statistics of the Kaggle *Mall Customer Segmentation* dataset.

### Session 02: K-Means Clustering Fundamentals
- **Task 1**: Manual Euclidean distance calculation and initial centroid assignment for food delivery 2D points `[[2,3], [5,8], [1,2], [6,9], [7,7]]`.
- **Task 2**: Implemented modular `assign_clusters(points, centroids)` function.
- **Task 3**: Expanded `assign_clusters(points, centroids, metric)` to support both `'euclidean'` and `'manhattan'` metrics.
- **Task 4**: Implemented `update_centroids(points, assignments, k)` to recompute mean vector coordinates for each cluster.
- **Task 5**: Performed Elbow Method on Zomato restaurant features, evaluating WCSS for $k \in [1, 6]$ and identifying optimal $k=3$.

### Session 03: Density-Based Clustering (DBSCAN)
- **Task 1**: Applied DBSCAN (`eps=0.35`, `min_samples=5`) on Mall Customers dataset, discovering 5 clusters and 1 noise point.
- **Task 2**: Color-coded visualization of DBSCAN clusters with noise points distinctly marked with black 'x' symbols.
- **Task 3**: Sensitivity grid analysis showing how altering `eps` and `min_samples` impacts noise points and cluster merging.
- **Task 4**: Side-by-side visual comparison of DBSCAN vs K-Means, highlighting DBSCAN's superior outlier isolation and non-convex cluster handling.
- **Task 5**: Executed DBSCAN on `make_moons` dataset with added noise points; extracted and printed exact outlier indices (`[2, 81, 97, 106, 144, 178, 191, 193, 223, 241, 292, 301, 302, 303, 304, 306, 307, 308]`).

### Session 04: Dimensionality Reduction (PCA) & Evaluation
- **Task 1**: Reduced Iris dataset from 4 features to 2 Principal Components (PC1 explains 72.96%, PC2 explains 22.85%, cumulative 95.81%).
- **Task 2**: Spotify-style 2D PCA scatter plot colored by Iris species (*Setosa*, *Versicolor*, *Virginica*).
- **Task 3**: Reduced 10-feature Zomato dataset using PCA and plotted cumulative variance curve across all 10 components.
- **Task 4**: Fitted K-Means ($k=3$) on 2D PCA Iris projection, plotting clusters and red 'X' centroids.
- **Task 5**: Evaluated clustering quality using `silhouette_score`, achieving a score of **0.5092**, confirming strong cluster separation.

### Session 05: Hierarchical Clustering Fundamentals & Distance Metrics
- **Task 1**: Computed full $6 \times 6$ pairwise Euclidean distance matrix step-by-step for 6 food delivery orders.
- **Task 2**: Implemented Ward linkage Agglomerative Hierarchical Clustering on an 8-song audio feature dataset (Tempo & Energy) and saved dendrogram visualization (`task_2_song_dendrogram.png`).
- **Task 3**: Generated side-by-side comparative dendrograms using `Single`, `Complete`, and `Average` linkage methods, documenting chaining effects and cluster compactness (`task_3_linkage_comparison.png`).
- **Task 4**: Created modular helper `calculate_distances(p1, p2)` returning both Euclidean and Manhattan metric values for 2D points.
- **Task 5**: Evaluated linkage methods for a "Similar Movies" recommender feature, recommending **Ward's linkage** for variance minimization and balanced cluster sizes.

### Session 06: Advanced Dendrogram Analysis & Iris Hierarchical Clustering
- **Task 1**: Built Ward linkage hierarchical clustering pipeline on the 4D Iris dataset (StandardScaler normalized) and rendered tree structure with cut height threshold (`task_1_iris_ward_dendrogram.png`).
- **Task 2**: Performed horizontal dendrogram truncation at height $h=10.0$ via `scipy.cluster.hierarchy.fcluster`, forming 3 clusters (Counts: Cluster 1: 49, Cluster 2: 71, Cluster 3: 30) and cross-tabulating against true Iris species labels.
- **Task 3**: Created a 3-panel comparative dendrogram visualization on Iris comparing `Ward`, `Single`, and `Complete` linkage methods (`task_3_linkage_dendrograms_comparison.png`).
- **Task 4**: Documented dendrogram interpretation guidelines explaining branch merge heights as dissimilarity, identifying most similar/distinct clusters, and choosing optimal horizontal tree cuts.

### Session 07: Customer Segmentation Tutorial Assignment
- **Task 1**: Loaded Kaggle Mall Customer Segmentation dataset (`mall_customers.csv`, 200 records x 5 features) and displayed first 10 rows.
- **Task 2**: Audited missing values in `Age`, `Annual Income (k$)`, and `Spending Score (1-100)` columns, confirming **0 missing values** (100% complete).
- **Task 3**: Plotted `Annual Income (k$)` vs `Spending Score (1-100)` scatterplot colored by gender (`task_3_customer_scatterplot.png`).
- **Task 4**: Generated a Seaborn correlation matrix heatmap across numerical features (`task_4_correlation_heatmap.png`); identified `Age` and `Spending Score (1-100)` as the strongest correlated domain feature pair ($r = +0.1153$).
- **Task 5**: Executed K-Means ($k=3$, `random_state=42`) using `Annual Income` and `Spending Score`, attached `Cluster` column to DataFrame (Cluster 0: 116 customers, Cluster 1: 40 customers, Cluster 2: 44 customers), and generated segment visualization with red 'X' centroids (`task_5_kmeans_customer_clusters.png`).


