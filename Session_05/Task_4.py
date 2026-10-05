"""
Session 05 - Task 4
Function: calculate_distances(p1, p2)
Returns both Euclidean and Manhattan distances between two 2D points (x1, y1) and (x2, y2).
Tested on 3 coordinate pairs.
"""

import math

def calculate_distances(p1, p2):
    """
    Calculates Euclidean and Manhattan distances between two 2D points.

    Parameters:
    - p1: tuple/list (x1, y1)
    - p2: tuple/list (x2, y2)

    Returns:
    - tuple: (euclidean_distance, manhattan_distance)
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Euclidean distance: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    euclidean = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Manhattan distance: |x2 - x1| + |y2 - y1|
    manhattan = abs(x2 - x1) + abs(y2 - y1)
    
    return euclidean, manhattan

if __name__ == "__main__":
    test_pairs = [
        ("Pair 1 (Order 1 vs Order 2)", (15, 2.5), (30, 6.0)),
        ("Pair 2 (Order 3 vs Order 4)", (12, 1.8), (45, 10.5)),
        ("Pair 3 (Order 5 vs Order 6)", (25, 5.0), (50, 12.0))
    ]
    
    print("=" * 80)
    print("SESSION 05 - TASK 4: Dual Distance Calculation (Euclidean & Manhattan)")
    print("=" * 80)
    print(f"{'Point Pair Description':<30} | {'Point 1':<12} | {'Point 2':<12} | {'Euclidean':<12} | {'Manhattan'}")
    print("-" * 80)
    
    for desc, p1, p2 in test_pairs:
        euc_dist, man_dist = calculate_distances(p1, p2)
        print(f"{desc:<30} | {str(p1):<12} | {str(p2):<12} | {euc_dist:<12.4f} | {man_dist:<12.4f}")
    print("=" * 80)
