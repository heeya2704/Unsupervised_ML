"""
Section B — Task 2: Order Dataset Preprocessor
M7-A1 | Missing values, one-hot encoding, MinMax scaling
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def build_raw_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            "restaurant_rating": [
                4.5,
                None,
                3.8,
                4.2,
                None,
                4.9,
                3.1,
                4.0,
                3.6,
                4.7,
                2.9,
                4.4,
            ],
            "payment_method": [
                "Cash",
                "Card",
                "Wallet",
                "Card",
                "Cash",
                "Wallet",
                "Card",
                "Cash",
                "Wallet",
                "Card",
                "Cash",
                "Wallet",
            ],
            "cuisine_type": [
                "Indian",
                "Chinese",
                "Italian",
                "Fast Food",
                "Indian",
                "Chinese",
                "Italian",
                "Fast Food",
                "Indian",
                "Chinese",
                "Italian",
                "Fast Food",
            ],
            "order_value": [
                220.50,
                415.00,
                680.25,
                150.00,
                310.75,
                540.00,
                95.40,
                275.00,
                890.10,
                360.00,
                199.99,
                425.50,
            ],
        }
    )


def main() -> None:
    df = build_raw_orders()
    print("=" * 72)
    print("  ORDER DATASET PREPROCESSOR")
    print("=" * 72)
    print("\n1) Raw data (before preprocessing)")
    print(df.to_string(index=False))
    print("\n   Missing values per column:")
    print(df.isnull().sum().to_string())

    rating_mean = df["restaurant_rating"].mean()
    df["restaurant_rating"] = df["restaurant_rating"].fillna(rating_mean)

    print(f"\n2) Filled restaurant_rating NaN with column mean = {rating_mean:.4f}")
    print("   Missing values after imputation:")
    print(df.isnull().sum().to_string())
    remaining = int(df.isnull().sum().sum())
    print(f"   Total remaining NaN values: {remaining}")
    if remaining != 0:
        raise RuntimeError("Expected zero missing values after imputation.")

    df = pd.get_dummies(
        df,
        columns=["payment_method", "cuisine_type"],
        dtype=int,
    )

    print("\n3) One-hot encoded payment_method and cuisine_type (original columns dropped)")

    scaler = MinMaxScaler()
    df[["restaurant_rating", "order_value"]] = scaler.fit_transform(
        df[["restaurant_rating", "order_value"]]
    )

    print("\n4) MinMaxScaler applied to restaurant_rating and order_value only")
    print("\n5) Final preprocessed DataFrame")
    print(df.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
