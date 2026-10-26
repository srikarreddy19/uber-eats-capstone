
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")
MODELS = Path("models")

for folder in [PROCESSED, FIGURES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(RAW / "customers.csv")
orders = pd.read_csv(RAW / "orders.csv", parse_dates=["order_timestamp"])
restaurants = pd.read_csv(RAW / "restaurants.csv")
deliveries = pd.read_csv(RAW / "deliveries.csv")
reviews = pd.read_csv(RAW / "reviews.csv")

# 1. Customer feature engineering
customer_orders = orders[orders["order_status"] == "Completed"].copy()
customer_orders["is_weekend"] = (
    customer_orders["order_timestamp"].dt.dayofweek >= 5
).astype(int)
customer_orders["is_late_night"] = (
    customer_orders["order_timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3, 4])
).astype(int)

customer_features = customer_orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    average_order_value=("order_amount", "mean"),
    total_spending=("order_amount", "sum"),
    first_order=("order_timestamp", "min"),
    last_order=("order_timestamp", "max"),
    weekend_orders=("is_weekend", "sum"),
    late_night_orders=("is_late_night", "sum")
).reset_index()

customer_features["ordering_frequency"] = (
    customer_features["total_orders"] /
    (((customer_features["last_order"] - customer_features["first_order"])
      .dt.days / 30).clip(lower=1))
)

customer_ratings = reviews.groupby("customer_id")["rating"].mean()
customer_features["average_rating_given"] = (
    customer_features["customer_id"].map(customer_ratings)
)

customer_features = customer_features.drop(
    columns=["first_order", "last_order"]
)
customer_features["average_rating_given"] = (
    customer_features["average_rating_given"].fillna(
        customer_features["average_rating_given"].median()
    )
)
customer_features.to_csv(
    PROCESSED / "customer_features.csv", index=False
)

# 2. Restaurant feature engineering
restaurant_orders = orders.merge(
    restaurants[["restaurant_id", "restaurant_rating",
                 "avg_preparation_time"]],
    on="restaurant_id", how="left"
)
restaurant_orders["avg_preparation_time"] = restaurant_orders["avg_preparation_time_y"]

restaurant_summary = restaurant_orders.groupby("restaurant_id").agg(
    total_orders=("order_id", "count"),
    revenue=("order_amount", lambda x: x[
        restaurant_orders.loc[x.index, "order_status"] == "Completed"
    ].sum()),
    cancellation_rate=("order_status", lambda x: (x == "Cancelled").mean()),
    average_rating=("restaurant_rating", "mean"),
    average_preparation_time=("avg_preparation_time", "mean")
).reset_index()

delivery_orders = deliveries.merge(
    orders[["order_id", "restaurant_id"]], on="order_id", how="left"
)
delivery_summary = delivery_orders.groupby("restaurant_id")[
    "delivery_time_minutes"
].mean()

restaurant_summary["average_delivery_time"] = (
    restaurant_summary["restaurant_id"].map(delivery_summary)
)
restaurant_summary["average_delivery_time"] = (
    restaurant_summary["average_delivery_time"].fillna(
        restaurant_summary["average_delivery_time"].median()
    )
)
restaurant_summary["average_rating"] = (
    restaurant_summary["average_rating"].fillna(
        restaurant_summary["average_rating"].median()
    )
)
restaurant_summary.to_csv(
    PROCESSED / "restaurant_features.csv", index=False
)

# 3. K-Means, DBSCAN, evaluation and PCA
def cluster_dataset(df, id_column, prefix):
    features = df.drop(columns=[id_column]).select_dtypes(
        include="number"
    ).copy()

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())
    features = features.loc[:, features.nunique() > 1]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Elbow method and silhouette scores
    inertias = []
    scores = {}
    max_k = min(6, len(df) - 1)

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        inertias.append(model.inertia_)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(scaled, labels)

    plt.figure()
    plt.plot(range(2, max_k + 1), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title(f"{prefix}: Elbow Method")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{prefix.lower()}_elbow.png")
    plt.close()

    best_k = max(scores, key=scores.get) if scores else 3
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled)

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled)

    if len(set(dbscan_labels) - {-1}) > 1:
        mask = dbscan_labels != -1
        dbscan_silhouette = silhouette_score(scaled[mask], dbscan_labels[mask])
    else:
        dbscan_silhouette = np.nan

    df[f"{prefix.lower()}_kmeans_cluster"] = kmeans_labels
    df[f"{prefix.lower()}_dbscan_cluster"] = dbscan_labels

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=kmeans_labels, s=15)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"{prefix}: K-Means Clusters")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{prefix.lower()}_pca.png")
    plt.close()

    # Cluster profiles use original, unscaled feature values
    profile = df.groupby(f"{prefix.lower()}_kmeans_cluster")[
        list(features.columns)
    ].mean().round(2)
    profile.to_csv(PROCESSED / f"{prefix.lower()}_cluster_profiles.csv")

    # Generic cluster names based on spending/revenue and activity
    if prefix == "Customer":
        activity_col = "total_orders"
        value_col = "total_spending"
    else:
        activity_col = "total_orders"
        value_col = "revenue"

    high_activity = profile[activity_col].median()
    high_value = profile[value_col].median()

    names = {}
    for cluster, row in profile.iterrows():
        if row[activity_col] >= high_activity and row[value_col] >= high_value:
            names[cluster] = "High-Value Frequent"
        elif row[value_col] < high_value:
            names[cluster] = "Lower-Value"
        else:
            names[cluster] = "High-Value Occasional"

    df[f"{prefix.lower()}_segment_name"] = (
        df[f"{prefix.lower()}_kmeans_cluster"].map(names)
    )
    df.to_csv(PROCESSED / f"{prefix.lower()}_segmented.csv", index=False)

    joblib.dump(scaler, MODELS / f"{prefix.lower()}_scaler.joblib")
    joblib.dump(kmeans, MODELS / f"{prefix.lower()}_kmeans.joblib")
    joblib.dump(dbscan, MODELS / f"{prefix.lower()}_dbscan.joblib")

    return best_k, scores, dbscan_silhouette, profile, names

customer_result = cluster_dataset(
    customer_features, "customer_id", "Customer"
)
restaurant_result = cluster_dataset(
    restaurant_summary, "restaurant_id", "Restaurant"
)

with open("reports/segmentation_results.txt", "w") as f:
    for name, result in [
        ("Customer", customer_result),
        ("Restaurant", restaurant_result)
    ]:
        best_k, scores, dbscan_score, profile, names = result
        f.write(f"\n{name} Segmentation\n")
        f.write(f"Selected K: {best_k}\n")
        f.write(f"Silhouette scores: {scores}\n")
        f.write(f"DBSCAN silhouette (excluding noise): {dbscan_score}\n")
        f.write(f"Business names: {names}\n")
        f.write(f"Cluster profile:\n{profile}\n")

print("Day 2 segmentation completed.")
print("Feature datasets saved in data/processed/")
print("Models saved in models/")
print("Plots saved in reports/figures/")
print("Results saved in reports/segmentation_results.txt")
