"""
Session 01 - Task 1
Generate a random dataset of 10 users with two features:
- daily_app_opens
- avg_session_duration_min
Print dataset as a list of dictionaries.
"""

import random

def generate_user_dataset(num_users=10, seed=42):
    random.seed(seed)
    dataset = []
    for user_id in range(1, num_users + 1):
        user_data = {
            "user_id": f"User_{user_id:02d}",
            "daily_app_opens": random.randint(1, 35),
            "avg_session_duration_min": round(random.uniform(0.5, 45.0), 2)
        }
        dataset.append(user_data)
    return dataset

if __name__ == "__main__":
    users = generate_user_dataset(num_users=10)
    print("=" * 60)
    print("SESSION 01 - TASK 1: Generated User Dataset (List of Dictionaries)")
    print("=" * 60)
    for user in users:
        print(user)
