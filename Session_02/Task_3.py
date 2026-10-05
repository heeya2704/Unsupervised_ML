"""
Session 02 - Task 3
Updated assign_clusters function supporting 'euclidean' and 'manhattan' distance metrics.
"""

import math

def assign_clusters(points, centroids, metric='euclidean'):
    """
    Assigns each point in 'points' to the nearest centroid in 'centroids'
    using either Euclidean or Manhattan distance.

    Parameters:
    - points: list of tuples/lists, e.g., [[x1, y1], [x2, y2], ...]
    - centroids: list of tuples/lists, e.g., [[c1_x, c1_y], [c2_x, c2_y], ...]
    - metric: 'euclidean' (default) or 'manhattan'

    Returns:
    - list of integers representing cluster index (0-indexed) for each point.
    """
    metric = metric.lower()
    if metric not in ['euclidean', 'manhattan']:
        raise ValueError(f"Unsupported metric '{metric}'. Choose 'euclidean' or 'manhattan'.")

    assignments = []
    
    for pt in points:
        min_dist = float('inf')
        best_cluster = 0
        
        for c_idx, c in enumerate(centroids):
            if metric == 'euclidean':
                dist = math.sqrt(sum((p_i - c_i) ** 2 for p_i, c_i in zip(pt, c)))
            elif metric == 'manhattan':
                dist = sum(abs(p_i - c_i) for p_i, c_i in zip(pt, c))
                
            if dist < min_dist:
                min_dist = dist
                best_cluster = c_idx
                
        assignments.append(best_cluster)
        
    return assignments

if __name__ == "__main__":
    sample_points = [[2, 3], [5, 8], [1, 2], [6, 9], [7, 7], [4, 5], [8, 2]]
    sample_centroids = [[2, 3], [6, 9]]
    
    euc_assignments = assign_clusters(sample_points, sample_centroids, metric='euclidean')
    man_assignments = assign_clusters(sample_points, sample_centroids, metric='manhattan')
    
    print("=" * 80)
    print("SESSION 02 - TASK 3: Cluster Assignment with Euclidean vs Manhattan Metric")
    print("=" * 80)
    print(f"Sample Points  : {sample_points}")
    print(f"Centroids      : {sample_centroids}\n")
    print(f"{'Point':<12} | {'Euclidean Cluster':<20} | {'Manhattan Cluster':<20} | Match?")
    print("-" * 75)
    for pt, e_cls, m_cls in zip(sample_points, euc_assignments, man_assignments):
        match_str = "YES" if e_cls == m_cls else "DIFFERENT (*)"
        print(f"{str(pt):<12} | Cluster {e_cls:<12} | Cluster {m_cls:<12} | {match_str}")
