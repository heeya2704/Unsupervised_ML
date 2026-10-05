"""
Section C — Mini Capstone: Food Delivery Customer Intelligence System
M7-A1 | Menu-driven console app (Modules 5, 6, and 7)

Options:
  1) View Statistical Summary
  2) Predict Cancellation Risk for a New Order
  3) Run Customer Segmentation
  4) Exit
"""

from __future__ import annotations

import statistics
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_orders(n: int = 200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order_value = rng.uniform(80, 1400, size=n)
    delivery_distance_km = rng.uniform(0.4, 20.0, size=n)
    restaurant_rating = rng.uniform(2.2, 5.0, size=n)

    # Right-skew delivery times: most 22–38 min, a few long delays
    delivery_time = rng.normal(30, 5.5, size=n)
    n_outliers = max(8, n // 20)
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    delivery_time[outlier_idx] = rng.uniform(78, 120, size=n_outliers)
    delivery_time = np.clip(delivery_time, 12, 130)

    night = rng.integers(0, 2, size=n)
    logit = (
        -2.2
        + 0.16 * delivery_distance_km
        - 0.001 * order_value
        + 0.85 * night
        - 0.5 * restaurant_rating
    )
    prob = 1 / (1 + np.exp(-logit))
    cancelled = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "order_value": np.round(order_value, 2),
            "delivery_distance_km": np.round(delivery_distance_km, 2),
            "restaurant_rating": np.round(restaurant_rating, 2),
            "delivery_time_min": np.round(delivery_time, 1),
            "cancelled": cancelled,
        }
    )


def build_customers(n: int = 180, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
    blocks = [
        np.column_stack(
            [
                rng.integers(1, 5, size=n1),
                rng.normal(175, 30, size=n1),
                rng.normal(3.5, 0.3, size=n1),
            ]
        ),
        np.column_stack(
            [
                rng.integers(6, 14, size=n2),
                rng.normal(310, 35, size=n2),
                rng.normal(4.15, 0.22, size=n2),
            ]
        ),
        np.column_stack(
            [
                rng.integers(15, 26, size=n3),
                rng.normal(600, 50, size=n3),
                rng.normal(4.65, 0.16, size=n3),
            ]
        ),
    ]
    data = np.vstack(blocks)
    rng.shuffle(data)
    df = pd.DataFrame(
        data,
        columns=["monthly_orders", "avg_order_value", "avg_delivery_rating"],
    )
    df["monthly_orders"] = df["monthly_orders"].clip(1, 40).round().astype(int)
    df["avg_order_value"] = df["avg_order_value"].clip(50, 900).round(2)
    df["avg_delivery_rating"] = df["avg_delivery_rating"].clip(1.0, 5.0).round(2)
    return df


def skew_label(mean_val: float, median_val: float, tolerance: float = 0.5) -> str:
    if mean_val > median_val + tolerance:
        return "right-skewed"
    if mean_val < median_val - tolerance:
        return "left-skewed"
    return "approximately symmetrical"


def print_column_stats(name: str, values: list[float], interpretation: str) -> None:
    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    stdev_val = statistics.stdev(values)
    direction = skew_label(mean_val, median_val)
    print(f"  Column: {name}")
    print(f"    Mean                 : {mean_val:8.2f}")
    print(f"    Median               : {median_val:8.2f}")
    print(f"    Standard deviation   : {stdev_val:8.2f}")
    print(f"    Skewness direction   : {direction}")
    print(f"    Interpretation       : {interpretation}")
    print()


def option_stats(orders: pd.DataFrame) -> None:
    print("\n--- Option 1: Statistical Summary ---")
    print_column_stats(
        "delivery_time_min",
        orders["delivery_time_min"].tolist(),
        "Most deliveries are typical, but a long right tail of delays "
        "pulls the mean above the median. Median is the better typical time.",
    )
    print_column_stats(
        "order_value",
        orders["order_value"].tolist(),
        "Order values spread widely around the centre. Compare mean and "
        "median to see whether a few large baskets dominate the average.",
    )


def train_cancel_model(orders: pd.DataFrame) -> RandomForestClassifier:
    features = ["order_value", "delivery_distance_km", "restaurant_rating"]
    model = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE)
    model.fit(orders[features], orders["cancelled"])
    return model


def prompt_float(label: str, minimum: float, maximum: float) -> Optional[float]:
    raw = input(f"  Enter {label} ({minimum} to {maximum}): ").strip()
    try:
        value = float(raw)
    except ValueError:
        print(f"  Error: '{raw}' is not a number. Please enter a numeric value.")
        return None
    if value < minimum or value > maximum:
        print(
            f"  Error: {label} must be between {minimum} and {maximum}. "
            f"You entered {value}."
        )
        return None
    return value


def option_predict(model: RandomForestClassifier) -> None:
    print("\n--- Option 2: Predict Cancellation Risk ---")
    print("  Enter order details. Invalid input returns you to the menu.")
    order_value = prompt_float("order_value", 1.0, 5000.0)
    if order_value is None:
        return
    distance = prompt_float("delivery_distance_km", 0.1, 50.0)
    if distance is None:
        return
    rating = prompt_float("restaurant_rating", 1.0, 5.0)
    if rating is None:
        return

    X = pd.DataFrame(
        [[order_value, distance, rating]],
        columns=["order_value", "delivery_distance_km", "restaurant_rating"],
    )
    proba_cancel = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    label = "High Cancellation Risk" if pred == 1 else "Low Cancellation Risk"
    print()
    print(f"  Result               : {label}")
    print(f"  P(cancelled)         : {proba_cancel:.1%}")
    print(f"  P(not cancelled)     : {1 - proba_cancel:.1%}")


def option_segment(customers: pd.DataFrame) -> None:
    print("\n--- Option 3: Customer Segmentation (K-Means, k=3) ---")
    features = ["monthly_orders", "avg_order_value", "avg_delivery_rating"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customers[features])
    model = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    counts = pd.Series(labels).value_counts().sort_index()
    centres = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=features,
    ).round(2)

    for cluster_id in range(3):
        print(f"  Cluster {cluster_id}: {int(counts.get(cluster_id, 0))} customers")
        row = centres.iloc[cluster_id]
        print(f"    Centroid monthly_orders      : {row['monthly_orders']:.2f}")
        print(f"    Centroid avg_order_value     : {row['avg_order_value']:.2f}")
        print(f"    Centroid avg_delivery_rating : {row['avg_delivery_rating']:.2f}")
        print()


def print_menu() -> None:
    print()
    print("=" * 56)
    print("  FOOD DELIVERY CUSTOMER INTELLIGENCE SYSTEM")
    print("=" * 56)
    print("  1) View Statistical Summary")
    print("  2) Predict Cancellation Risk for a New Order")
    print("  3) Run Customer Segmentation")
    print("  4) Exit")
    print("=" * 56)


def main() -> None:
    orders = build_orders()
    customers = build_customers()
    cancel_model = train_cancel_model(orders)
    print("Loaded in-memory order dataset and customer dataset.")
    print("Cancellation classifier trained in-session (Random Forest).")

    while True:
        print_menu()
        choice = input("  Select an option (1-4): ").strip()
        if choice == "1":
            option_stats(orders)
        elif choice == "2":
            option_predict(cancel_model)
        elif choice == "3":
            option_segment(customers)
        elif choice == "4":
            print("\n  Exiting. Thank you.")
            break
        else:
            print("  Error: please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
