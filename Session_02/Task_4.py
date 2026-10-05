"""
Session 02 - Task 4
Function: update_centroids(points, assignments, k)
Calculates the new centroid for each cluster as the mean of the assigned points.
"""

def update_centroids(points, assignments, k):
    """
    Calculates the new centroid for each cluster k as the mean of the assigned points.

    Parameters:
    - points: list of numerical coordinate lists/tuples, e.g., [[x1, y1], [x2, y2], ...]
    - assignments: list of integers indicating cluster label (0 to k-1) for each point.
    - k: integer, number of clusters.

    Returns:
    - list of updated centroid coordinates [[c1_x, c1_y, ...], [c2_x, c2_y, ...], ...]
    """
    num_features = len(points[0]) if points else 0
    new_centroids = []
    
    for cluster_id in range(k):
        # Extract points assigned to cluster_id
        cluster_points = [pt for pt, label in zip(points, assignments) if label == cluster_id]
        
        if cluster_points:
            # Calculate mean across each feature dimension
            centroid = [
                sum(pt[dim] for pt in cluster_points) / len(cluster_points)
                for dim in range(num_features)
            ]
        else:
            # Handle empty cluster fallback (keep dummy/zero or random point)
            centroid = [0.0] * num_features
            
        new_centroids.append(centroid)
        
    return new_centroids

if __name__ == "__main__":
    sample_points = [[2, 3], [5, 8], [1, 2], [6, 9], [7, 7]]
    sample_assignments = [0, 1, 0, 1, 1]  # Assigned to cluster 0 or 1
    num_clusters = 2
    
    old_centroids = [[2.0, 3.0], [6.0, 9.0]]
    updated_centroids = update_centroids(sample_points, sample_assignments, num_clusters)
    
    print("=" * 75)
    print("SESSION 02 - TASK 4: Centroid Update Step (K-Means)")
    print("=" * 75)
    print(f"Sample Points      : {sample_points}")
    print(f"Cluster Assignments: {sample_assignments}")
    print(f"Old Centroids      : {old_centroids}\n")
    
    for c_id in range(num_clusters):
        c_pts = [sample_points[i] for i, a in enumerate(sample_assignments) if a == c_id]
        print(f"Cluster {c_id+1}:")
        print(f"  • Points assigned: {c_pts}")
        print(f"  • Updated Centroid (Mean): {updated_centroids[c_id]}")
        print("-" * 75)
