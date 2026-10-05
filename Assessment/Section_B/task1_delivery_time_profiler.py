"""
Section B — Task 1: Delivery Time Statistical Profiler
M7-A1 | Food delivery domain
"""

import statistics

# At least 15 values: typical 20–40 min deliveries plus 3+ outliers above 75 min
delivery_times = [
    22, 25, 27, 28, 29, 30, 30, 30, 31, 32, 33, 34, 35, 36, 38, 40,
    82, 91, 105,
]


def skew_direction(mean_val: float, median_val: float, tolerance: float = 0.5) -> str:
    if mean_val > median_val + tolerance:
        return "right-skewed (positively skewed)"
    if mean_val < median_val - tolerance:
        return "left-skewed (negatively skewed)"
    return "approximately symmetrical"


def main() -> None:
    mean_val = statistics.mean(delivery_times)
    median_val = statistics.median(delivery_times)
    mode_val = statistics.mode(delivery_times)
    variance_val = statistics.variance(delivery_times)
    stdev_val = statistics.stdev(delivery_times)
    skew = skew_direction(mean_val, median_val)

    print("=" * 52)
    print("  DELIVERY TIME STATISTICAL PROFILER")
    print("  Food Delivery Platform - Operations Report")
    print("=" * 52)
    print(f"  Sample size (n)          : {len(delivery_times):>10}")
    print(f"  Mean (minutes)           : {mean_val:>10.2f}")
    print(f"  Median (minutes)         : {median_val:>10.2f}")
    print(f"  Mode (minutes)           : {mode_val:>10.2f}")
    print(f"  Variance                 : {variance_val:>10.2f}")
    print(f"  Standard deviation       : {stdev_val:>10.2f}")
    print("-" * 52)
    print("  Skewness conclusion")
    print("-" * 52)
    print(f"  Mean vs median           : {mean_val:.2f} vs {median_val:.2f}")
    print(f"  Distribution shape       : {skew}")
    print()
    print("  Most deliveries sit in the 20-40 minute range, but a")
    print("  few extreme delays (82, 91, 105 min) pull the mean")
    print("  above the median. The app should display the median")
    print("  as the typical delivery time, not the mean.")
    print("=" * 52)


if __name__ == "__main__":
    main()
