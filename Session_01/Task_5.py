"""
Session 01 - Task 5
Real-World Dataset Analysis for Customer Segmentation
Dataset: Mall Customer Segmentation Data (Kaggle / UCI)
"""

import pandas as pd
import os

def analyze_dataset(file_path="Datasets/mall_customers.csv"):
    dataset_info = {
        "Dataset Name": "Mall Customer Segmentation Data",
        "Repository Link": "https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python",
        "Target Application": "Customer Segmentation / Clustering",
        "Is Labeled or Unlabeled": "Unlabeled (Contains no ground-truth target variable; designed for unsupervised clustering)"
    }
    
    print("=" * 85)
    print("SESSION 01 - TASK 5: Real-World Dataset Structure & Overview")
    print("=" * 85)
    for k, v in dataset_info.items():
        print(f"• {k:<25}: {v}")
        
    print("-" * 85)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"\nDataset Shape: {df.shape[0]} rows (customers), {df.shape[1]} features (columns)")
        print("\nFeature Summary & Data Types:")
        print("-" * 85)
        
        info_df = pd.DataFrame({
            "Feature Name": df.columns,
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Sample Value": [df[col].iloc[0] for col in df.columns],
            "Description": [
                "Unique identifier for each store customer",
                "Categorical demographic feature (Male/Female)",
                "Continuous numerical feature (Age in years, 18-70)",
                "Continuous numerical feature (Annual income in thousands of dollars)",
                "Discrete score assigned by mall based on customer behavior & purchasing nature (1-100)"
            ]
        })
        print(info_df.to_string(index=False))
        print("\nSummary Statistics:")
        print("-" * 85)
        print(df.describe())
    else:
        print(f"File not found at {file_path}")

if __name__ == "__main__":
    analyze_dataset()
