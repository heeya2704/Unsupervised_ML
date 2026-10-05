"""
Session 05 - Task 1
Initial Euclidean Distance Matrix for 6 Food Delivery Orders.
Features: delivery_time (minutes) and distance (km).
"""

import math
import pandas as pd
import numpy as np

# 6 Food Delivery Orders: [delivery_time_min, distance_km]
orders = {
    "Order 1": [15, 2.5],
    "Order 2": [30, 6.0],
    "Order 3": [12, 1.8],
    "Order 4": [45, 10.5],
    "Order 5": [25, 5.0],
    "Order 6": [50, 12.0]
}

def compute_distance_matrix():
    order_names = list(orders.keys())
    n = len(order_names)
    dist_matrix = np.zeros((n, n))
    
    print("=" * 80)
    print("SESSION 05 - TASK 1: Food Delivery Orders - Initial Distance Matrix")
    print("=" * 80)
    print("Dataset of 6 Orders:")
    for name, coords in orders.items():
        print(f"  • {name}: Delivery Time = {coords[0]} min | Distance = {coords[1]} km")
    print("-" * 80)
    
    print("\nPairwise Euclidean Distance Calculations:")
    print("-" * 80)
    
    for i in range(n):
        for j in range(n):
            name_i, coords_i = order_names[i], orders[order_names[i]]
            name_j, coords_j = order_names[j], orders[order_names[j]]
            
            # Euclidean distance: sqrt((x2 - x1)^2 + (y2 - y1)^2)
            d = math.sqrt((coords_j[0] - coords_i[0])**2 + (coords_j[1] - coords_i[1])**2)
            dist_matrix[i, j] = round(d, 4)
            
            if i < j:
                print(f"dist({name_i}, {name_j}) = sqrt(({coords_j[0]}-{coords_i[0]})^2 + ({coords_j[1]}-{coords_i[1]})^2) = {d:.4f}")
                
    df_dist = pd.DataFrame(dist_matrix, index=order_names, columns=order_names)
    
    print("\n" + "=" * 80)
    print("FINAL 6x6 EUCLIDEAN DISTANCE MATRIX:")
    print("=" * 80)
    print(df_dist.to_string())
    print("=" * 80)
    return df_dist

if __name__ == "__main__":
    compute_distance_matrix()
