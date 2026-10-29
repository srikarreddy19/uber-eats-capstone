
import os
import re
import pandas as pd

os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def load_csv(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

# 1. Load outputs from previous days
customers = load_csv("data/processed/customer_segmented.csv")
restaurants = load_csv("data/processed/restaurant_segmented.csv")
reviews = load_csv("data/processed/cleaned_reviews.csv")
delivery_models = load_csv("data/processed/delivery_model_comparison.csv")
delivery_importance = load_csv("data/processed/delivery_feature_importance.csv")
delivery_features = load_csv("data/processed/delivery_time_features.csv")
demand_models = load_csv("data/processed/demand_model_comparison.csv")
forecast = load_csv("data/processed/final_demand_forecast.csv")

insights = []

def add_insight(area, finding, evidence, recommendation):
    insights.append({
        "area": area,
        "finding": finding,
        "evidence": evidence,
        "recommendation": recommendation
    })

# 2. Customer segmentation
if not customers.empty and "customer_segment_name" in customers.columns:
    customer_summary = customers.groupby(
        "customer_segment_name"
    ).agg(
        customers=("customer_id", "nunique"),
        average_spending=("total_spending", "mean"),
        average_orders=("total_orders", "mean"),
        total_segment_spending=("total_spending", "sum")
    ).sort_values("total_segment_spending", ascending=False)

    customer_summary.to_csv(
        "data/processed/day6_customer_segment_summary.csv"
    )

    top = customer_summary.iloc[0]
    add_insight(
        "Customer segmentation",
        f"{customer_summary.index[0]} contributes the highest total spending.",
        f"{top['customers']:.0f} customers; average spending "
        f"{top['average_spending']:.2f}; segment spending "
        f"{top['total_segment_spending']:.2f}.",
        "Prioritize retention initiatives for high-spending segments "
        "and test targeted offers before wider rollout."
    )

    if len(customer_summary) > 1:
        low = customer_summary.iloc[-1]
        add_insight(
            "Customer segmentation",
            f"{customer_summary.index[-1]} has the lowest total segment spending.",
            f"Average spending {low['average_spending']:.2f}; "
            f"average orders {low['average_orders']:.2f}.",
            "Investigate ordering frequency and customer feedback before "
            "testing offers designed to increase repeat orders."
        )

# 3. Restaurant segmentation
if not restaurants.empty and "restaurant_segment_name" in restaurants.columns:
    restaurant_summary = restaurants.groupby(
        "restaurant_segment_name"
    ).agg(
        restaurants=("restaurant_id", "nunique"),
        total_revenue=("revenue", "sum"),
        average_rating=("average_rating", "mean"),
        average_cancellation_rate=("cancellation_rate", "mean"),
        average_preparation_time=("average_preparation_time", "mean"),
        average_delivery_time=("average_delivery_time", "mean")
    ).sort_values("total_revenue", ascending=False)

    restaurant_summary.to_csv(
        "data/processed/day6_restaurant_segment_summary.csv"
    )

    intervention = restaurant_summary.sort_values(
        ["average_rating", "average_cancellation_rate"],
        ascending=[True, False]
    ).iloc[0]

    add_insight(
        "Restaurant segmentation",
        f"{restaurant_summary.index[
            restaurant_summary.index.get_loc(intervention.name)
        ]} is the segment to investigate first for service improvement.",
        f"Average rating {intervention['average_rating']:.2f}; "
        f"cancellation rate {intervention['average_cancellation_rate']:.1%}; "
        f"average delivery time {intervention['average_delivery_time']:.2f} minutes.",
        "Review restaurant-level ratings, cancellations, preparation times, "
        "and delivery times within this segment before deciding on interventions."
    )

# 4. Sentiment analysis and negative experience themes
if not reviews.empty and "sentiment" in reviews.columns:
    sentiment_counts = reviews["sentiment"].value_counts()
    sentiment_pct = (
        reviews["sentiment"].value_counts(normalize=True) * 100
    )
    sentiment_summary = pd.DataFrame({
        "review_count": sentiment_counts,
        "percentage": sentiment_pct
    })
    sentiment_summary.to_csv(
        "data/processed/day6_sentiment_summary.csv"
    )

    negative = reviews[
        reviews["sentiment"].astype(str).str.lower() == "negative"
    ].copy()

    text_col = (
        "review_text" if "review_text" in negative.columns
        else "cleaned_review"
    )
    text = negative[text_col].fillna("").str.lower()

    themes = {
        "Delivery delays": r"\blate\b|\bdelay(?:ed|s)?\b|\bslow\b|\bwait(?:ed|ing)?\b",
        "Cold food": r"\bcold\b|\btemperature\b",
        "Food quality": r"\bsoggy\b|\bstale\b|\bbad taste\b|\bundercooked\b|\bovercooked\b",
        "Missing or incorrect items": r"\bmissing\b|\bwrong\b|\bincorrect\b|\bforgot\b",
        "Driver experience": r"\bdriver\b|\brider\b|\bcourier\b"
    }

    theme_rows = []
    for theme, pattern in themes.items():
        count = int(text.str.contains(pattern, regex=True).sum())
        theme_rows.append({
            "theme": theme,
            "negative_reviews_matching": count
        })

    theme_df = pd.DataFrame(theme_rows).sort_values(
        "negative_reviews_matching", ascending=False
    )
    theme_df.to_csv(
        "data/processed/day6_negative_review_themes.csv", index=False
    )

    if len(negative) > 0 and not theme_df.empty:
        top_theme = theme_df.iloc[0]
        add_insight(
            "Customer experience",
            f"{top_theme['theme']} is the most frequent of the predefined "
            "negative-review themes.",
            f"{int(top_theme['negative_reviews_matching'])} of "
            f"{len(negative)} negative reviews match this theme. Themes can overlap.",
            "Inspect representative reviews and related order records, then "
            "test a targeted service improvement and monitor ratings."
        )

# 5. Delivery prediction model performance and feature importance
if not delivery_models.empty and "MAE" in delivery_models.columns:
    best_delivery = delivery_models.sort_values("MAE").iloc[0]
    delivery_models.to_csv(
        "data/processed/day6_delivery_model_summary.csv", index=False
    )

    add_insight(
        "Delivery prediction",
        f"{best_delivery['model']} has the lowest recorded test MAE.",
        f"MAE {best_delivery['MAE']:.2f} minutes; "
        f"RMSE {best_delivery['RMSE']:.2f} minutes; "
        f"R-squared {best_delivery['R2']:.3f}.",
        "Use predicted delivery times as planning estimates and track "
        "prediction errors on new orders before operational deployment."
    )

if not delivery_importance.empty:
    delivery_importance = delivery_importance.sort_values(
        "importance", ascending=False
    )
    delivery_importance.to_csv(
        "data/processed/day6_delivery_feature_importance.csv",
        index=False
    )
    top_features = delivery_importance.head(3)

    add_insight(
        "Delivery drivers",
        "The leading model features are: " +
        ", ".join(top_features["feature"].astype(str)),
        "; ".join(
            f"{row['feature']}: {row['importance']:.3f}"
            for _, row in top_features.iterrows()
        ),
        "Prioritize these factors in operational monitoring. Feature importance "
        "indicates predictive contribution, not proof of causation."
    )

# Check whether tip prediction exists
tip_model_exists = any(
    os.path.exists(path) for path in [
        "models/best_tip_model.joblib",
        "models/tip_prediction_model.joblib"
    ]
)
if not tip_model_exists:
    add_insight(
        "Tip prediction",
        "A saved tip prediction model was not found in the expected locations.",
        "Day 4 outputs contain delivery-time prediction results, not verified "
        "tip-model evaluation.",
        "Treat tip prediction as a follow-up task: verify tip data, build a "
        "chronological or held-out regression evaluation, and compare MAE/RMSE."
    )

# 6. Demand forecasting and capacity planning
if not demand_models.empty and "MAE" in demand_models.columns:
    best_demand = demand_models.sort_values("MAE").iloc[0]
    add_insight(
        "Demand forecasting",
        f"{best_demand['model']} has the lowest recorded forecasting MAE.",
        f"MAE {best_demand['MAE']:.3f} orders per hour; "
        f"RMSE {best_demand['RMSE']:.3f}; "
        f"nonzero-only MAPE {best_demand['MAPE_nonzero_pct']:.2f}%.",
        "Use the forecast as an initial planning signal, but validate its "
        "accuracy against future actual demand before making staffing changes."
    )

if not forecast.empty and "final_forecast" in forecast.columns:
    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
    forecast = forecast.dropna(subset=["ds", "final_forecast"])
    forecast["hour"] = forecast["ds"].dt.hour
    forecast["day_of_week"] = forecast["ds"].dt.day_name()

    peak_hour = forecast.groupby("hour")["final_forecast"].mean().idxmax()
    peak_day = forecast.groupby("day_of_week")["final_forecast"].sum().idxmax()
    total_forecast = forecast["final_forecast"].sum()

    forecast.to_csv(
        "data/processed/day6_forecast_with_time_features.csv", index=False
    )

    add_insight(
        "Capacity planning",
        f"The forecast's highest average hourly demand is at {int(peak_hour):02d}:00.",
        f"Forecast horizon: {len(forecast)} hours; "
        f"forecast total: {total_forecast:.0f} orders; "
        f"largest forecast daily total: {peak_day}.",
        "Consider additional rider availability around forecast peak periods. "
        "Use zone-level forecasts and live demand data before assigning capacity."
    )

# 7. Integrated outputs
insight_df = pd.DataFrame(insights)
insight_df.to_csv(
    "data/processed/day6_integrated_business_insights.csv", index=False
)

with open("reports/day6_business_insights.md", "w") as f:
    f.write("# Day 6 – Integrated ML Pipeline and Business Insights\n\n")
    f.write("## Pipeline overview\n\n")
    f.write(
        "Raw datasets → existing cleaning and feature engineering → "
        "customer/restaurant segmentation → sentiment analysis → "
        "delivery-time prediction → demand forecasting → evaluation → "
        "business insights.\n\n"
    )
    f.write("## Evidence-based findings and recommendations\n\n")

    for i, row in insight_df.iterrows():
        f.write(f"### {i + 1}. {row['area']}\n\n")
        f.write(f"**Finding:** {row['finding']}\n\n")
        f.write(f"**Evidence:** {row['evidence']}\n\n")
        f.write(f"**Recommendation:** {row['recommendation']}\n\n")

    f.write("## Model limitations\n\n")
    f.write(
        "- The datasets are synthetic; results may not represent real Uber Eats operations.\n"
        "- Customer and restaurant cluster labels are data-derived and require "
        "business validation.\n"
        "- Review themes use predefined keyword matching and may miss context, "
        "sarcasm, or overlapping complaints.\n"
        "- Sentiment labels are based on the existing training pipeline and may "
        "reflect rating-derived labels rather than independently verified sentiment.\n"
        "- Delivery feature importance shows predictive association, not causation.\n"
        "- Demand forecasting has relatively high nonzero-only MAPE; low-volume "
        "hours can make percentage errors unstable.\n"
        "- Tip prediction is not reported as completed without a verified model "
        "and evaluation results.\n"
        "- Forecasting and prediction performance should be rechecked on new, "
        "chronologically later data.\n"
    )

    f.write("\n## Future improvements\n\n")
    f.write(
        "- Validate insights with real production data and business stakeholders.\n"
        "- Add zone-level demand forecasting and event-aware features.\n"
        "- Tune forecasting models and monitor errors over time.\n"
        "- Build and evaluate a dedicated tip prediction model if reliable tip data exists.\n"
        "- Automate data validation, model versioning, and scheduled retraining.\n"
        "- Track business KPIs such as cancellations, on-time delivery, ratings, "
        "repeat orders, and revenue after interventions.\n"
    )

print("\nDay 6 integration completed.")
print("Insights generated:", len(insight_df))
print("\nKey outputs:")
print("- data/processed/day6_integrated_business_insights.csv")
print("- data/processed/day6_customer_segment_summary.csv")
print("- data/processed/day6_restaurant_segment_summary.csv")
print("- data/processed/day6_sentiment_summary.csv")
print("- data/processed/day6_negative_review_themes.csv")
print("- data/processed/day6_delivery_model_summary.csv")
print("- data/processed/day6_delivery_feature_importance.csv")
print("- data/processed/day6_forecast_with_time_features.csv")
print("- reports/day6_business_insights.md")
print("\nReview the report before submission; findings depend on available data.")
