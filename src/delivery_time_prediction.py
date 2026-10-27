from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

for folder in [OUT, REPORTS, FIGURES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(RAW / "orders.csv")
deliveries = pd.read_csv(RAW / "deliveries.csv")
restaurants = pd.read_csv(RAW / "restaurants.csv")

df = orders.merge(deliveries, on="order_id", how="inner")
restaurant_cols = restaurants[
    ["restaurant_id", "category", "restaurant_rating"]
]
df = df.merge(restaurant_cols, on="restaurant_id", how="left")

df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
df["order_hour"] = df["order_timestamp"].dt.hour
df["day_of_week"] = df["order_timestamp"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

segment_file = OUT / "customer_segmented.csv"
if segment_file.exists():
    segments = pd.read_csv(segment_file)
    cluster_col = "customer_kmeans_cluster"
    if "customer_id" in segments.columns and cluster_col in segments.columns:
        segments = segments[["customer_id", cluster_col]].drop_duplicates("customer_id")
        df = df.merge(segments, on="customer_id", how="left")

target = "delivery_time_minutes"
numeric = [
    "delivery_distance_km", "order_amount", "num_items",
    "avg_preparation_time", "restaurant_rating",
    "order_hour", "day_of_week", "is_weekend"
]
categorical = ["traffic_condition", "weather", "category"]

if "customer_kmeans_cluster" in df.columns:
    categorical.append("customer_kmeans_cluster")

numeric = [c for c in numeric if c in df.columns]
categorical = [c for c in categorical if c in df.columns]

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])
df = df[df[target] > 0].copy()

features = numeric + categorical
df[features + [target]].to_csv(
    OUT / "delivery_time_features.csv", index=False
)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric),
    ("categorical", categorical_pipe, categorical)
])

models = {
    "Baseline Median": DummyRegressor(strategy="median"),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=15,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
}

results = []
fitted = {}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    cv_mae = -cross_val_score(
        pipeline, X_train, y_train, cv=3,
        scoring="neg_mean_absolute_error", n_jobs=1
    ).mean()

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_MAE_3fold": cv_mae
    })
    fitted[name] = pipeline
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}, CV MAE={cv_mae:.2f}")

comparison = pd.DataFrame(results).sort_values("MAE")
comparison.to_csv(OUT / "delivery_model_comparison.csv", index=False)

best_name = comparison.iloc[0]["model"]
best_model = fitted[best_name]
joblib.dump(best_model, MODELS / "best_delivery_time_model.joblib")

forest = fitted["Random Forest"]
importance = forest.named_steps["model"].feature_importances_
feature_names = forest.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)
importance_df.to_csv(OUT / "delivery_feature_importance.csv", index=False)

best_predictions = best_model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y, bins=30)
axes[0].set_title("Delivery Time Distribution")
axes[0].set_xlabel("Delivery time (minutes)")
axes[0].set_ylabel("Number of deliveries")

axes[1].scatter(df["delivery_distance_km"], df[target], alpha=0.3)
axes[1].set_title("Distance vs Delivery Time")
axes[1].set_xlabel("Distance (km)")
axes[1].set_ylabel("Delivery time (minutes)")
plt.tight_layout()
plt.savefig(FIGURES / "delivery_time_eda.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_predictions, alpha=0.5)
limits = [
    min(y_test.min(), best_predictions.min()),
    max(y_test.max(), best_predictions.max())
]
plt.plot(limits, limits, linestyle="--")
plt.xlabel("Actual delivery time (minutes)")
plt.ylabel("Predicted delivery time (minutes)")
plt.title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig(FIGURES / "delivery_actual_vs_predicted.png", dpi=150)
plt.close()

with open(REPORTS / "delivery_model_report.txt", "w") as f:
    f.write("DAY 4 - DELIVERY TIME PREDICTION\n\n")
    f.write(f"Rows used: {len(df)}\n")
    f.write(f"Features used: {', '.join(features)}\n\n")
    f.write(comparison.to_string(index=False))
    f.write(f"\n\nRecommended model by lowest test MAE: {best_name}\n")
    f.write(f"Test MAE: {comparison.iloc[0]['MAE']:.3f} minutes\n")
    f.write("\nFeature importance is calculated from the Random Forest model.\n")
    f.write("Synthetic data results may not generalize to real deliveries.\n")

print("\nDay 4 completed.")
print(f"Rows used: {len(df)}")
print(f"Recommended model: {best_name}")
print("Saved feature dataset, model comparison, feature importance, model, report and plots.")
