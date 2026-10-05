"""
Session 07 - Task 2
Missing Value Check for 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)'.
Calculates total missing values and missing percentage for each column.
"""

import os
import pandas as pd

def check_missing_values():
    possible_paths = [
        "Datasets/mall_customers.csv",
        "../Datasets/mall_customers.csv",
        "c:/Users/heeya/OneDrive/Documents/TOPS/Machine Learning/Unsupervised_ML/Assignment/Datasets/mall_customers.csv"
    ]
    
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break
            
    df = pd.read_csv(file_path)
    
    # Specified target columns to evaluate
    target_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    missing_counts = df[target_cols].isnull().sum()
    total_rows = len(df)
    
    print("=" * 85)
    print("SESSION 07 - TASK 2: Missing Values Audit")
    print("=" * 85)
    print(f"{'Target Column Name':<30} | {'Missing Count':<15} | {'Missing Percentage':<20}")
    print("-" * 85)
    
    for col in target_cols:
        count = missing_counts[col]
        pct = (count / total_rows) * 100
        print(f"{col:<30} | {count:<15} | {pct:<20.2f}%")
        
    print("-" * 85)
    print(f"Total Missing Values across specified columns: {missing_counts.sum()}")
    print(f"Total Missing Values across entire DataFrame : {df.isnull().sum().sum()}")
    print("=" * 85)

if __name__ == "__main__":
    check_missing_values()
