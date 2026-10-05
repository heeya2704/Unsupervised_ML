"""
Section B — Task 3: Order Cancellation Classifier with Evaluation
M7-A1 | Random Forest, stratified split, classification report, confusion matrix
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_RECORDS = 160
CANCEL_RATE = 0.22


def make_order_dataset(n: int = N_RECORDS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    order_value = rng.uniform(80, 1200, size=n)
    delivery_distance_km = rng.uniform(0.5, 18.0, size=n)
    hour_of_day = rng.integers(0, 24, size=n)
    restaurant_rating = rng.uniform(2.0, 5.0, size=n)

    # Higher cancel probability: far, cheap, late-night, low rating
    night = ((hour_of_day >= 22) | (hour_of_day <= 5)).astype(float)
    logit = (
        -2.4
        + 0.18 * delivery_distance_km
        - 0.0012 * order_value
        + 0.9 * night
        - 0.55 * restaurant_rating
    )
    prob = 1 / (1 + np.exp(-logit))
    cancelled = (rng.random(n) < prob).astype(int)

    # Nudge class balance toward 20–25% cancelled
    current = cancelled.mean()
    if current < 0.18 or current > 0.28:
        n_pos = int(round(CANCEL_RATE * n))
        scores = prob
        top = np.argsort(scores)[-n_pos:]
        cancelled = np.zeros(n, dtype=int)
        cancelled[top] = 1

    return pd.DataFrame(
        {
            "order_value": np.round(order_value, 2),
            "delivery_distance_km": np.round(delivery_distance_km, 2),
            "hour_of_day": hour_of_day,
            "restaurant_rating": np.round(restaurant_rating, 2),
            "cancelled": cancelled,
        }
    )


def labelled_confusion_matrix(y_true, y_pred) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print("Confusion matrix (rows = actual, columns = predicted)")
    print()
    print("                 Predicted 0      Predicted 1")
    print(f"  Actual 0 (kept)     TN={tn:<6}     FP={fp:<6}")
    print(f"  Actual 1 (cancel)   FN={fn:<6}     TP={tp:<6}")
    print()
    print("  Explicit counts")
    print(f"    True Positives  (TP) : {tp}  - cancelled orders correctly flagged")
    print(f"    True Negatives  (TN) : {tn}  - completed orders correctly identified")
    print(f"    False Positives (FP) : {fp}  - completed orders wrongly flagged")
    print(f"    False Negatives (FN) : {fn}  - cancelled orders missed")


def main() -> None:
    df = make_order_dataset()
    cancel_pct = 100 * df["cancelled"].mean()

    print("=" * 72)
    print("  ORDER CANCELLATION CLASSIFIER")
    print("=" * 72)
    print(f"  Records              : {len(df)}")
    print(f"  Cancelled share      : {cancel_pct:.1f}% (target ~20-25%)")
    print(f"  Features             : order_value, delivery_distance_km,")
    print("                         hour_of_day, restaurant_rating")

    X = df[["order_value", "delivery_distance_km", "hour_of_day", "restaurant_rating"]]
    y = df["cancelled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n  Train size:", len(X_train), " | Test size:", len(X_test))
    print("\n" + "-" * 72)
    print("  Classification report")
    print("-" * 72)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["not cancelled (0)", "cancelled (1)"],
            digits=3,
        )
    )
    print("-" * 72)
    labelled_confusion_matrix(y_test, y_pred)
    print("=" * 72)


if __name__ == "__main__":
    main()
