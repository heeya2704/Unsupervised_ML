import os
import pandas as pd
import numpy as np

os.makedirs("Datasets", exist_ok=True)

# 1. Create Mall Customers Dataset (200 samples)
np.random.seed(42)
n_samples = 200

customer_ids = list(range(1, n_samples + 1))
genders = np.random.choice(['Male', 'Female'], size=n_samples)
ages = np.random.randint(18, 71, size=n_samples)

cluster_centers = [
    (20, 20),   # Low income, Low spending
    (25, 78),   # Low income, High spending
    (55, 50),   # Medium income, Medium spending
    (88, 18),   # High income, Low spending
    (86, 82)    # High income, High spending
]

incomes = []
spending_scores = []

for i in range(n_samples):
    center = cluster_centers[i % len(cluster_centers)]
    inc = int(np.clip(np.random.normal(center[0], 6), 15, 137))
    sp = int(np.clip(np.random.normal(center[1], 8), 1, 99))
    incomes.append(inc)
    spending_scores.append(sp)

df_mall = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Annual Income (k$)': incomes,
    'Spending Score (1-100)': spending_scores
})

df_mall.to_csv("Datasets/mall_customers.csv", index=False)
print("Created Datasets/mall_customers.csv")

# 2. Create Zomato Ratings Dataset (150 samples, 10 numerical features)
np.random.seed(101)
n_zomato = 150

avg_cost = np.random.randint(150, 2500, size=n_zomato)
rating = np.round(np.random.uniform(2.5, 4.9, size=n_zomato), 1)
votes = np.random.randint(20, 5000, size=n_zomato)
delivery_time_min = np.random.randint(15, 60, size=n_zomato)
discount_percent = np.random.choice([0, 10, 15, 20, 30, 40, 50], size=n_zomato)
cuisines_count = np.random.randint(1, 8, size=n_zomato)
distance_km = np.round(np.random.uniform(0.5, 12.0, size=n_zomato), 1)
seating_capacity = np.random.randint(10, 120, size=n_zomato)
online_orders_count = np.random.randint(5, 1200, size=n_zomato)
reviews_count = np.random.randint(2, 450, size=n_zomato)

df_zomato = pd.DataFrame({
    'average_cost': avg_cost,
    'rating': rating,
    'votes': votes,
    'delivery_time_min': delivery_time_min,
    'discount_percent': discount_percent,
    'cuisines_count': cuisines_count,
    'distance_km': distance_km,
    'seating_capacity': seating_capacity,
    'online_orders_count': online_orders_count,
    'reviews_count': reviews_count
})

df_zomato.to_csv("Datasets/zomato_ratings.csv", index=False)
print("Created Datasets/zomato_ratings.csv")
