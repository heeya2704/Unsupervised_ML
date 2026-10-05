"""
Session 07 - Task 1
Load Customer Segmentation Tutorial Dataset (Mall Customers) using Pandas.
Displays dataset shape, schema info, and first 10 rows.
"""

import os
import pandas as pd

def load_dataset():
    # Resolve relative filepath from root workspace or script directory
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
            
    if not file_path:
        raise FileNotFoundError("Could not locate 'mall_customers.csv' in Datasets directory.")
        
    df = pd.read_csv(file_path)
    
    print("=" * 85)
    print("SESSION 07 - TASK 1: Customer Segmentation Dataset Loader")
    print("=" * 85)
    print(f"Dataset Path : {file_path}")
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("-" * 85)
    print("\nColumn Header Names & Data Types:")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"  • {col:<30} : {dtype}")
        
    print("\n" + "=" * 85)
    print("FIRST 10 ROWS OF MALL CUSTOMER DATASET:")
    print("=" * 85)
    print(df.head(10).to_string(index=False))
    print("=" * 85)
    
    return df

if __name__ == "__main__":
    load_dataset()
