
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 1000
N_RESTAURANTS = 100
N_DRIVERS = 200
N_ORDERS = 10000

cities = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai"]
categories = ["Indian", "Chinese", "Italian", "Fast Food", "Desserts", "Healthy"]
weather_types = ["Clear", "Rainy", "Cloudy", "Windy"]
traffic_types = ["Low", "Medium", "High"]
payment_methods = ["UPI", "Card", "Cash", "Wallet"]
order_statuses = ["Completed", "Cancelled", "Failed"]


def save_dataset(df, filename):
    df.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"{filename}: {len(df)} rows, {len(df.columns)} columns")


# 1. Customers
customers = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)],
    "age": np.random.randint(18, 66, N_CUSTOMERS),
    "gender": np.random.choice(["Male", "Female", "Other"], N_CUSTOMERS),
    "city": np.random.choice(cities, N_CUSTOMERS),
    "signup_date": pd.to_datetime("2024-01-01") +
                   pd.to_timedelta(
                       np.random.randint(0, 730, N_CUSTOMERS), unit="D"
                   )
})

save_dataset(customers, "customers.csv")


# 2. Restaurants
restaurants = pd.DataFrame({
    "restaurant_id": [f"R{i:03d}" for i in range(1, N_RESTAURANTS + 1)],
    "restaurant_name": [f"Restaurant_{i}" for i in range(1, N_RESTAURANTS + 1)],
    "category": np.random.choice(categories, N_RESTAURANTS),
    "city": np.random.choice(cities, N_RESTAURANTS),
    "restaurant_rating": np.round(np.random.uniform(2.8, 5.0, N_RESTAURANTS), 1),
    "avg_preparation_time": np.random.randint(10, 41, N_RESTAURANTS)
})

save_dataset(restaurants, "restaurants.csv")


# 3. Drivers
drivers = pd.DataFrame({
    "driver_id": [f"D{i:04d}" for i in range(1, N_DRIVERS + 1)],
    "city": np.random.choice(cities, N_DRIVERS),
    "vehicle_type": np.random.choice(
        ["Bike", "Scooter", "Car"], N_DRIVERS, p=[0.3, 0.6, 0.1]
    ),
    "driver_rating": np.round(np.random.uniform(3.0, 5.0, N_DRIVERS), 1),
    "experience_years": np.random.randint(0, 11, N_DRIVERS)
})

save_dataset(drivers, "drivers.csv")


# 4. Orders
order_times = pd.to_datetime("2026-01-01") + pd.to_timedelta(
    np.random.randint(0, 180 * 24 * 60, N_ORDERS), unit="m"
)

orders = pd.DataFrame({
    "order_id": [f"O{i:06d}" for i in range(1, N_ORDERS + 1)],
    "customer_id": np.random.choice(customers["customer_id"], N_ORDERS),
    "restaurant_id": np.random.choice(restaurants["restaurant_id"], N_ORDERS),
    "order_timestamp": order_times,
    "order_amount": np.round(np.random.gamma(4, 180, N_ORDERS), 2),
    "num_items": np.random.randint(1, 9, N_ORDERS),
    "order_status": np.random.choice(
        order_statuses, N_ORDERS, p=[0.88, 0.09, 0.03]
    )
})

orders = orders.merge(
    restaurants[["restaurant_id", "avg_preparation_time", "city"]],
    on="restaurant_id",
    how="left"
)

orders = orders.rename(columns={"city": "restaurant_city"})

save_dataset(orders, "orders.csv")


# 5. Deliveries: one record for each completed order
completed_orders = orders[orders["order_status"] == "Completed"].copy()
n_deliveries = len(completed_orders)

deliveries = pd.DataFrame({
    "order_id": completed_orders["order_id"].values,
    "driver_id": np.random.choice(drivers["driver_id"], n_deliveries),
    "delivery_distance_km": np.round(
        np.random.uniform(0.5, 15, n_deliveries), 2
    ),
    "traffic_condition": np.random.choice(
        traffic_types, n_deliveries, p=[0.3, 0.45, 0.25]
    ),
    "weather": np.random.choice(
        weather_types, n_deliveries, p=[0.5, 0.2, 0.25, 0.05]
    )
})

traffic_delay = deliveries["traffic_condition"].map({
    "Low": 0, "Medium": 8, "High": 18
})

weather_delay = deliveries["weather"].map({
    "Clear": 0, "Cloudy": 2, "Windy": 4, "Rainy": 12
})

deliveries["delivery_time_minutes"] = (
    10
    + deliveries["delivery_distance_km"] * 3
    + completed_orders["avg_preparation_time"].to_numpy()
    + traffic_delay.to_numpy()
    + weather_delay.to_numpy()
    + np.random.normal(0, 4, n_deliveries)
).round().clip(10).astype(int)

deliveries["delivery_time_minutes"] = deliveries[
    "delivery_time_minutes"
].fillna(10)

save_dataset(deliveries, "deliveries.csv")


# 6. Payments: one record per order
payments = orders[["order_id", "order_amount", "order_status"]].copy()

payments["payment_method"] = np.random.choice(
    payment_methods, len(payments), p=[0.5, 0.25, 0.1, 0.15]
)

payments["tip_amount"] = np.where(
    payments["order_status"] == "Completed",
    np.round(
        payments["order_amount"] * np.random.uniform(0.02, 0.20, len(payments)),
        2
    ),
    0
)

payments["payment_status"] = np.where(
    payments["order_status"] == "Completed",
    "Paid",
    np.where(payments["order_status"] == "Cancelled", "Refunded", "Failed")
)

save_dataset(payments, "payments.csv")


# 7. Reviews: a sample of completed orders
review_pool = completed_orders.sample(
    frac=0.65, random_state=42
).copy()

review_delivery_times = review_pool["order_id"].map(
    deliveries.set_index("order_id")["delivery_time_minutes"]
)

review_ratings = np.where(
    review_delivery_times > 55,
    np.random.choice([1, 2, 3], len(review_pool), p=[0.5, 0.3, 0.2]),
    np.random.choice([3, 4, 5], len(review_pool), p=[0.15, 0.35, 0.5])
)

positive_reviews = [
    "Great food and quick delivery!",
    "Delicious meal, very happy with the service.",
    "Excellent experience, everything was perfect.",
    "Fresh food and friendly delivery service.",
    "Loved the food and packaging."
]

negative_reviews = [
    "Food arrived late and cold.",
    "Very disappointing, delivery took too long.",
    "The order was delayed and food was soggy.",
    "Poor service and a long wait.",
    "Not happy with the late delivery."
]

neutral_reviews = [
    "The food was okay.",
    "Average experience, nothing special.",
    "Order was acceptable.",
    "Food was fine, but could be better."
]

review_texts = []
for rating in review_ratings:
    if rating <= 2:
        review_texts.append(np.random.choice(negative_reviews))
    elif rating >= 4:
        review_texts.append(np.random.choice(positive_reviews))
    else:
        review_texts.append(np.random.choice(neutral_reviews))

reviews = pd.DataFrame({
    "review_id": [f"V{i:06d}" for i in range(1, len(review_pool) + 1)],
    "order_id": review_pool["order_id"].values,
    "customer_id": review_pool["customer_id"].values,
    "restaurant_id": review_pool["restaurant_id"].values,
    "rating": review_ratings,
    "review_text": review_texts,
    "review_timestamp": review_pool["order_timestamp"].values
})

save_dataset(reviews, "reviews.csv")


# Data quality practice: introduce a small number of missing values
for filename, column in [
    ("customers.csv", "age"),
    ("restaurants.csv", "restaurant_rating"),
    ("reviews.csv", "review_text")
]:
    filepath = OUTPUT_DIR / filename
    df = pd.read_csv(filepath)

    n_missing = max(1, int(len(df) * 0.01))
    missing_indices = np.random.choice(df.index, n_missing, replace=False)
    df.loc[missing_indices, column] = np.nan

    df.to_csv(filepath, index=False)

print("\nAll seven synthetic datasets generated successfully.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
