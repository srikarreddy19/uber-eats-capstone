# Day 6 – Integrated ML Pipeline and Business Insights

## Pipeline overview

Raw datasets → existing cleaning and feature engineering → customer/restaurant segmentation → sentiment analysis → delivery-time prediction → demand forecasting → evaluation → business insights.

## Evidence-based findings and recommendations

### 1. Customer segmentation

**Finding:** High-Value Frequent contributes the highest total spending.

**Evidence:** 420 customers; average spending 8461.42; segment spending 3553797.34.

**Recommendation:** Prioritize retention initiatives for high-spending segments and test targeted offers before wider rollout.

### 2. Customer segmentation

**Finding:** Lower-Value has the lowest total segment spending.

**Evidence:** Average spending 4881.22; average orders 6.93.

**Recommendation:** Investigate ordering frequency and customer feedback before testing offers designed to increase repeat orders.

### 3. Restaurant segmentation

**Finding:** High-Value Frequent is the segment to investigate first for service improvement.

**Evidence:** Average rating 3.90; cancellation rate 8.1%; average delivery time 71.68 minutes.

**Recommendation:** Review restaurant-level ratings, cancellations, preparation times, and delivery times within this segment before deciding on interventions.

### 4. Customer experience

**Finding:** Delivery delays is the most frequent of the predefined negative-review themes.

**Evidence:** 2752 of 3462 negative reviews match this theme. Themes can overlap.

**Recommendation:** Inspect representative reviews and related order records, then test a targeted service improvement and monitor ratings.

### 5. Delivery prediction

**Finding:** Ridge Regression has the lowest recorded test MAE.

**Evidence:** MAE 3.24 minutes; RMSE 4.09 minutes; R-squared 0.953.

**Recommendation:** Use predicted delivery times as planning estimates and track prediction errors on new orders before operational deployment.

### 6. Delivery drivers

**Finding:** The leading model features are: numeric__delivery_distance_km, numeric__avg_preparation_time, categorical__traffic_condition_High

**Evidence:** numeric__delivery_distance_km: 0.529; numeric__avg_preparation_time: 0.248; categorical__traffic_condition_High: 0.085

**Recommendation:** Prioritize these factors in operational monitoring. Feature importance indicates predictive contribution, not proof of causation.

### 7. Tip prediction

**Finding:** A saved tip prediction model was not found in the expected locations.

**Evidence:** Day 4 outputs contain delivery-time prediction results, not verified tip-model evaluation.

**Recommendation:** Treat tip prediction as a follow-up task: verify tip data, build a chronological or held-out regression evaluation, and compare MAE/RMSE.

### 8. Demand forecasting

**Finding:** Prophet has the lowest recorded forecasting MAE.

**Evidence:** MAE 1.208 orders per hour; RMSE 1.523; nonzero-only MAPE 51.04%.

**Recommendation:** Use the forecast as an initial planning signal, but validate its accuracy against future actual demand before making staffing changes.

### 9. Capacity planning

**Finding:** The forecast's highest average hourly demand is at 22:00.

**Evidence:** Forecast horizon: 168 hours; forecast total: 389 orders; largest forecast daily total: Monday.

**Recommendation:** Consider additional rider availability around forecast peak periods. Use zone-level forecasts and live demand data before assigning capacity.

## Model limitations

- The datasets are synthetic; results may not represent real Uber Eats operations.
- Customer and restaurant cluster labels are data-derived and require business validation.
- Review themes use predefined keyword matching and may miss context, sarcasm, or overlapping complaints.
- Sentiment labels are based on the existing training pipeline and may reflect rating-derived labels rather than independently verified sentiment.
- Delivery feature importance shows predictive association, not causation.
- Demand forecasting has relatively high nonzero-only MAPE; low-volume hours can make percentage errors unstable.
- Tip prediction is not reported as completed without a verified model and evaluation results.
- Forecasting and prediction performance should be rechecked on new, chronologically later data.

## Future improvements

- Validate insights with real production data and business stakeholders.
- Add zone-level demand forecasting and event-aware features.
- Tune forecasting models and monitor errors over time.
- Build and evaluate a dedicated tip prediction model if reliable tip data exists.
- Automate data validation, model versioning, and scheduled retraining.
- Track business KPIs such as cancellations, on-time delivery, ratings, repeat orders, and revenue after interventions.
