
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# 1. Prepare hourly demand
orders = pd.read_csv("data/raw/orders.csv")
orders["order_timestamp"] = pd.to_datetime(
    orders["order_timestamp"], errors="coerce"
)
orders = orders.dropna(subset=["order_timestamp", "order_id"])
orders = orders.drop_duplicates(subset=["order_id"])
orders = orders.sort_values("order_timestamp")

hourly = (
    orders.set_index("order_timestamp")
    .resample("h")["order_id"]
    .count()
    .asfreq("h", fill_value=0)
    .rename("order_count")
    .reset_index()
    .rename(columns={"order_timestamp": "ds"})
)

hourly.to_csv("data/processed/hourly_demand.csv", index=False)

if len(hourly) < 100:
    raise ValueError("Not enough hourly data to build reliable forecasts.")

print("Hourly records:", len(hourly))
print("Date range:", hourly["ds"].min(), "to", hourly["ds"].max())

# 2. Explore hourly, daily and weekend patterns
hourly["hour"] = hourly["ds"].dt.hour
hourly["day_of_week"] = hourly["ds"].dt.day_name()
hourly["is_weekend"] = (hourly["ds"].dt.dayofweek >= 5).astype(int)

hourly.groupby("hour")["order_count"].mean().to_csv(
    "data/processed/demand_by_hour.csv"
)
hourly.groupby("day_of_week")["order_count"].mean().to_csv(
    "data/processed/demand_by_day.csv"
)
hourly.groupby("is_weekend")["order_count"].mean().to_csv(
    "data/processed/weekend_demand.csv"
)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
hourly.set_index("ds")["order_count"].plot(ax=axes[0], title="Hourly Order Demand")
hourly.groupby("hour")["order_count"].mean().plot(
    kind="bar", ax=axes[1], title="Average Demand by Hour"
)
hourly.groupby("is_weekend")["order_count"].mean().plot(
    kind="bar", ax=axes[2], title="Weekday (0) vs Weekend (1)"
)
plt.tight_layout()
plt.savefig("reports/figures/demand_eda.png")
plt.close()

peak_hour = int(hourly.groupby("hour")["order_count"].mean().idxmax())
peak_day = hourly.groupby("day_of_week")["order_count"].mean().idxmax()

# 3. Decomposition: trend, seasonality and residual noise
series = hourly.set_index("ds")["order_count"].astype(float)

decomposition = seasonal_decompose(
    series, model="additive", period=24, extrapolate_trend="freq"
)
decomposition_df = pd.DataFrame({
    "ds": series.index,
    "observed": decomposition.observed.values,
    "trend": decomposition.trend.values,
    "seasonal": decomposition.seasonal.values,
    "residual": decomposition.resid.values
})
decomposition_df.to_csv(
    "data/processed/demand_decomposition.csv", index=False
)

fig = decomposition.plot()
fig.set_size_inches(12, 9)
fig.tight_layout()
fig.savefig("reports/figures/demand_decomposition.png")
plt.close(fig)

# 4. Stationarity: ADF test before and after differencing
adf_original = adfuller(series)
adf_diff = adfuller(series.diff().dropna())

with open("reports/adf_test_results.txt", "w") as f:
    f.write(f"Original series ADF statistic: {adf_original[0]:.4f}\n")
    f.write(f"Original series p-value: {adf_original[1]:.6f}\n")
    f.write(f"Original stationary (p < 0.05): {adf_original[1] < 0.05}\n\n")
    f.write(f"First-differenced ADF statistic: {adf_diff[0]:.4f}\n")
    f.write(f"First-differenced p-value: {adf_diff[1]:.6f}\n")
    f.write(f"Differenced stationary (p < 0.05): {adf_diff[1] < 0.05}\n")

# 5. Chronological train/test split (no shuffling)
split = int(len(hourly) * 0.8)
train = hourly.iloc[:split].copy()
test = hourly.iloc[split:].copy()

train_series = train.set_index("ds")["order_count"].astype(float)
test_series = test.set_index("ds")["order_count"].astype(float)

actual = test_series.values
results = []
fitted_models = {}

# ARIMA model
try:
    arima_train = ARIMA(
        train_series, order=(2, 1, 2)
    ).fit()
except Exception:
    arima_train = ARIMA(train_series, order=(1, 1, 1)).fit()

arima_predictions = np.maximum(
    0, np.asarray(arima_train.forecast(steps=len(test)))
)
fitted_models["ARIMA"] = arima_train

# Prophet model
prophet_train = train[["ds", "order_count"]].rename(
    columns={"order_count": "y"}
)
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)
prophet_model.add_country_holidays(country_name="India")
prophet_model.fit(prophet_train)

prophet_test_dates = pd.DataFrame({"ds": test["ds"]})
prophet_predictions = np.maximum(
    0, prophet_model.predict(prophet_test_dates)["yhat"].values
)
fitted_models["Prophet"] = prophet_model

# 6. Compare forecasting models
def calculate_metrics(y_true, y_pred):
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) /
                       y_true[nonzero])) * 100
        if nonzero.any() else np.nan
    )
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE_nonzero_pct": mape
    }

for name, predictions in [
    ("ARIMA", arima_predictions),
    ("Prophet", prophet_predictions)
]:
    results.append({
        "model": name,
        **calculate_metrics(actual, predictions)
    })

comparison = pd.DataFrame(results).sort_values("MAE")
comparison.to_csv("data/processed/demand_model_comparison.csv", index=False)

predictions_df = pd.DataFrame({
    "ds": test["ds"].values,
    "actual": actual,
    "ARIMA": arima_predictions,
    "Prophet": prophet_predictions
})
predictions_df.to_csv(
    "data/processed/demand_test_predictions.csv", index=False
)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(test["ds"], actual, label="Actual", linewidth=1)
ax.plot(test["ds"], arima_predictions, label="ARIMA", alpha=0.8)
ax.plot(test["ds"], prophet_predictions, label="Prophet", alpha=0.8)
ax.set_title("Hourly Demand: Actual vs Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Orders per hour")
ax.legend()
fig.tight_layout()
fig.savefig("reports/figures/demand_model_comparison.png")
plt.close(fig)

# 7. Save full-data models and forecast the next 7 days
best_model_name = comparison.iloc[0]["model"]

arima_full = ARIMA(series, order=(2, 1, 2)).fit()
arima_full.save("models/arima_demand_model.pkl")
joblib.dump(prophet_model, "models/prophet_demand_model.joblib")

future_dates = pd.date_range(
    start=hourly["ds"].max() + pd.Timedelta(hours=1),
    periods=24 * 7,
    freq="h"
)

arima_future = np.maximum(
    0, np.asarray(arima_full.forecast(steps=len(future_dates)))
)

prophet_full_data = hourly[["ds", "order_count"]].rename(
    columns={"order_count": "y"}
)
prophet_full = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)
prophet_full.add_country_holidays(country_name="India")
prophet_full.fit(prophet_full_data)
prophet_future = prophet_full.predict(pd.DataFrame({"ds": future_dates}))

joblib.dump(prophet_full, "models/prophet_demand_model.joblib")

forecast = pd.DataFrame({
    "ds": future_dates,
    "ARIMA_forecast": arima_future,
    "Prophet_forecast": np.maximum(0, prophet_future["yhat"].values),
    "Prophet_lower": np.maximum(0, prophet_future["yhat_lower"].values),
    "Prophet_upper": np.maximum(0, prophet_future["yhat_upper"].values)
})

forecast["selected_model"] = best_model_name
forecast["final_forecast"] = forecast[
    f"{best_model_name}_forecast"
]
forecast.to_csv("data/processed/final_demand_forecast.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(series.index[-24 * 7:], series.values[-24 * 7:],
        label="Recent actual demand")
ax.plot(forecast["ds"], forecast["final_forecast"],
        label=f"7-day forecast ({best_model_name})")
if best_model_name == "Prophet":
    ax.fill_between(
        forecast["ds"], forecast["Prophet_lower"],
        forecast["Prophet_upper"], alpha=0.2,
        label="Prophet uncertainty interval"
    )
ax.set_title("Next 7 Days: Hourly Demand Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Orders per hour")
ax.legend()
fig.tight_layout()
fig.savefig("reports/figures/final_demand_forecast.png")
plt.close(fig)

# 8. Business interpretation
with open("reports/demand_forecast_report.txt", "w") as f:
    f.write("UBER EATS HOURLY DEMAND FORECASTING\n\n")
    f.write(f"Historical start: {hourly['ds'].min()}\n")
    f.write(f"Historical end: {hourly['ds'].max()}\n")
    f.write(f"Hourly observations: {len(hourly)}\n")
    f.write(f"Training observations: {len(train)}\n")
    f.write(f"Test observations: {len(test)}\n\n")
    f.write("Model comparison (chronological test set):\n")
    f.write(comparison.to_string(index=False))
    f.write(f"\n\nSelected model by lowest MAE: {best_model_name}\n")
    f.write(f"Highest average demand hour: {peak_hour}:00\n")
    f.write(f"Highest average demand weekday: {peak_day}\n")
    f.write(f"Forecast horizon: {len(forecast)} hours (7 days)\n")
    f.write(
        f"Expected total orders over forecast horizon: "
        f"{forecast['final_forecast'].sum():.0f}\n\n"
    )
    f.write(
        "Business interpretation:\n"
        f"- Prioritize staffing and delivery capacity around {peak_hour}:00, "
        "the highest average-demand hour in the historical data.\n"
        f"- Review demand patterns on {peak_day} when planning restaurant "
        "and driver availability.\n"
        "- Use the hourly forecast to plan staffing and operational capacity.\n"
        "- Forecasts are estimates based on synthetic historical orders; "
        "validate against real demand before operational use.\n"
        "- MAPE is calculated only where actual demand is nonzero.\n"
    )

print("\nModel comparison:")
print(comparison.to_string(index=False))
print("\nBest model:", best_model_name)
print("Peak demand hour:", peak_hour)
print("Peak day:", peak_day)
print("Forecast orders over next 7 days:",
      round(forecast["final_forecast"].sum()))
print("\nDay 5 completed. Deliverables saved in data/processed, models, and reports.")

