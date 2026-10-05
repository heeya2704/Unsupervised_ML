"""
Session 02 - Task 2
Function: assign_clusters(points, centroids) using Euclidean Distance.
"""

import math

def assign_clusters(points, centroids):
    """
    Assigns each point in 'points' to the nearest centroid in 'centroids'
    based on Euclidean distance.

    Parameters:
    - points: list of tuples/lists, e.g., [[x1, y1], [x2, y2], ...]
    - centroids: list of tuples/lists, e.g., [[c1_x, c1_y], [c2_x, c2_y], ...]

    Returns:
    - list of integers representing cluster index (0-indexed) for each point.
    """
    assignments = []
    
    for pt in points:
        min_dist = float('inf')
        best_cluster = 0
        
        for c_idx, c in enumerate(centroids):
            # Calculate Euclidean distance: sqrt(sum((p_i - c_i)^2))
            dist = math.sqrt(sum((p_i - c_i) ** 2 for p_i, c_i in zip(pt, c)))
            if dist < min_dist:
                min_dist = dist
                best_cluster = c_idx
                
        assignments.append(best_cluster)
        
    return assignments

if __name__ == "__main__":
    sample_points = [[2, 3], [5, 8], [1, 2], [6, 9], [7, 7], [3, 4], [8, 2]]
    sample_centroids = [[2, 3], [6, 9]]
    
    cluster_labels = assign_clusters(sample_points, sample_centroids)
    
    print("=" * 70)
    print("SESSION 02 - TASK 2: Euclidean Distance Cluster Assignment")
    print("=" * 70)
    print(f"Input Points   : {sample_points}")
    print(f"Input Centroids: {sample_centroids}")
    print(f"Assignments    : {cluster_labels}\n")
    
    for pt, label in zip(sample_points, cluster_labels):
        print(f"Point {pt} -> Assigned to Cluster {label} (Centroid {sample_centroids[label]})")
