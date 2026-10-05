"""
Session 01 - Task 4
3 Real-World Scenarios for Unsupervised Learning (Zomato, Flipkart, Spotify)
Describes required data and insights discovered.
"""

scenarios = [
    {
        "platform": "Zomato",
        "use_case": "Restaurant & Dining Behavior Segmentation",
        "description": "Grouping restaurants into hyper-local culinary hubs and dining profiles to optimize delivery radius and targeted promotions.",
        "data_needed": [
            "Order timestamps and peak ordering hours",
            "Average order value (AOV) and price range",
            "Cuisine tags and top ordered dishes",
            "Delivery distance, fulfillment time, and user rating distribution"
        ],
        "insights_discovered": [
            "Identifies 'Late-Night Comfort Food' spots vs 'Healthy Office Lunch' vendors.",
            "Detects underserved geographical zones with high demand for specific cuisines.",
            "Helps Zomato customize homepage banners and coupon discounts based on user cluster preferences."
        ]
    },
    {
        "platform": "Flipkart",
        "use_case": "Customer Purchasing Power & Shopping Persona Clustering",
        "description": "Clustering shoppers based on spending patterns, device category affinity, and price sensitivity.",
        "data_needed": [
            "Monthly transaction volume and total spend",
            "Product category breakdown (Electronics vs Apparel vs Groceries)",
            "Return rate, discount usage frequency, and payment mode preferences (COD vs EMI vs Credit Card)",
            "Session search queries and item view-to-cart ratio"
        ],
        "insights_discovered": [
            "Segments users into 'Deal Hunters' (high discount sensitivity), 'Tech Enthusiasts' (early adopters), and 'Bargain Household Buyers'.",
            "Enables personalized promotional pushes and dynamic offer structuring during Big Billion Days sales."
        ]
    },
    {
        "platform": "Spotify",
        "use_case": "Song Acoustic Clustering & Micro-Genre Discovery",
        "description": "Grouping tracks by raw audio attributes and sonic characteristics without relying on manual genre tags.",
        "data_needed": [
            "Audio signals: Tempo (BPM), Valence (musical positiveness), Danceability, Acousticness, Energy",
            "Track duration, loudness, and instrumentalness ratio",
            "User playback skip frequency and completion rates"
        ],
        "insights_discovered": [
            "Discovers niche micro-genres (e.g., 'Chillhop Focus', 'Dark Synthwave Workout').",
            "Powers Spotify's 'Daily Mix' and 'Discover Weekly' by matching songs with similar acoustic vectors."
        ]
    }
]

def display_scenarios():
    print("=" * 85)
    print("SESSION 01 - TASK 4: Real-World Unsupervised Learning Scenarios")
    print("=" * 85)
    for idx, s in enumerate(scenarios, 1):
        print(f"\nScenario {idx}: {s['platform']} - {s['use_case']}")
        print("-" * 85)
        print(f"Overview: {s['description']}")
        print("\nData Needed:")
        for item in s['data_needed']:
            print(f"  • {item}")
        print("\nKey Insights Discovered:")
        for insight in s['insights_discovered']:
            print(f"  • {insight}")
        print("=" * 85)

if __name__ == "__main__":
    display_scenarios()
