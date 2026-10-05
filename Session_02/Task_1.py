"""
Session 02 - Task 1
Food delivery location manual clustering (K-Means Step 1: Initialization & Assignment)
Points: [2,3], [5,8], [1,2], [6,9], [7,7]
Distance Metric: Euclidean Distance
"""

import math
import matplotlib.pyplot as plt

def run_task_1():
    # 2D points representing food delivery locations
    points = [[2, 3], [5, 8], [1, 2], [6, 9], [7, 7]]
    point_names = [f"P{i+1} {pts}" for i, pts in enumerate(points)]
    
    # Randomly/explicitly select 2 initial centroids
    centroids = [[2, 3], [6, 9]]  # C1 and C2
    
    print("=" * 75)
    print("SESSION 02 - TASK 1: Food Delivery Location Clustering (Step 1)")
    print("=" * 75)
    print(f"Dataset Points: {points}")
    print(f"Initial Centroids chosen: C1 = {centroids[0]}, C2 = {centroids[1]}\n")
    print("Step-by-Step Euclidean Distance Calculations:")
    print("-" * 75)
    
    assignments = []
    
    for idx, (x, y) in enumerate(points):
        d_c1 = math.sqrt((x - centroids[0][0])**2 + (y - centroids[0][1])**2)
        d_c2 = math.sqrt((x - centroids[1][0])**2 + (y - centroids[1][1])**2)
        
        assigned_cluster = 0 if d_c1 <= d_c2 else 1
        assignments.append(assigned_cluster)
        
        print(f"Point P{idx+1} [{x}, {y}]:")
        print(f"  • Distance to C1 {centroids[0]}: sqrt(({x}-{centroids[0][0]})^2 + ({y}-{centroids[0][1]})^2) = {d_c1:.4f}")
        print(f"  • Distance to C2 {centroids[1]}: sqrt(({x}-{centroids[1][0]})^2 + ({y}-{centroids[1][1]})^2) = {d_c2:.4f}")
        print(f"  => Assigned to Cluster {assigned_cluster + 1} (Centroid C{assigned_cluster + 1})\n")
        
    print("Final Initial Cluster Assignments:")
    for cluster_id in range(2):
        c_points = [points[i] for i, a in enumerate(assignments) if a == cluster_id]
        print(f"  Cluster {cluster_id + 1} (Centroid {centroids[cluster_id]}): {c_points}")

    # Plot visualization
    plt.figure(figsize=(7, 6))
    colors = ['blue', 'green']
    
    for idx, (x, y) in enumerate(points):
        c_idx = assignments[idx]
        plt.scatter(x, y, color=colors[c_idx], s=120, zorder=3)
        plt.annotate(f"P{idx+1}({x},{y})", (x, y), textcoords="offset points", xytext=(8, -4), fontsize=9)
        
    for c_idx, (cx, cy) in enumerate(centroids):
        plt.scatter(cx, cy, color='red', marker='X', s=200, label=f'Initial Centroid C{c_idx+1} ({cx},{cy})', zorder=4)
        
    plt.title("Food Delivery Locations & Initial Cluster Assignment", fontsize=12, fontweight='bold')
    plt.xlabel("X Coordinate (Km East)", fontsize=10)
    plt.ylabel("Y Coordinate (Km North)", fontsize=10)
    plt.xlim(0, 9)
    plt.ylim(0, 11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("Session_02/task_1_delivery_clusters.png", dpi=300)
    print("\nSaved plot visualization to Session_02/task_1_delivery_clusters.png")
    plt.close()

if __name__ == "__main__":
    run_task_1()
