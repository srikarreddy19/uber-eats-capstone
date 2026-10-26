# Uber Eats Operational & Customer Intelligence System

## Project Overview
This project uses synthetic Uber Eats-style data to analyze customer segments, restaurant performance, delivery operations, customer reviews, and order demand.

## 1. Customers — `customers.csv`

| Column | Description |
|---|---|
| customer_id | Unique customer identifier |
| age | Customer age |
| gender | Customer gender |
| city | Customer's city |
| signup_date | Date the customer registered |

## 2. Restaurants — `restaurants.csv`

| Column | Description |
|---|---|
| restaurant_id | Unique restaurant identifier |
| restaurant_name | Restaurant name |
| category | Cuisine or food category |
| city | Restaurant location |
| restaurant_rating | Restaurant rating |
| avg_preparation_time | Average food preparation time in minutes |

## 3. Drivers — `drivers.csv`

| Column | Description |
|---|---|
| driver_id | Unique driver identifier |
| city | Driver's city |
| vehicle_type | Driver's vehicle type |
| driver_rating | Driver rating |
| experience_years | Years of driving experience |

## 4. Orders — `orders.csv`

| Column | Description |
|---|---|
| order_id | Unique order identifier |
| customer_id | Customer who placed the order |
| restaurant_id | Restaurant receiving the order |
| order_timestamp | Date and time of the order |
| order_amount | Total order amount |
| num_items | Number of items ordered |
| order_status | Completed, Cancelled, or Failed |
| avg_preparation_time | Restaurant's average preparation time in minutes |
| restaurant_city | City where the restaurant is located |

## 5. Deliveries — `deliveries.csv`

| Column | Description |
|---|---|
| order_id | Associated order identifier |
| driver_id | Driver assigned to the delivery |
| delivery_distance_km | Delivery distance in kilometres |
| traffic_condition | Traffic level during delivery |
| weather | Weather condition during delivery |
| delivery_time_minutes | Total delivery time in minutes |

## 6. Payments — `payments.csv`

| Column | Description |
|---|---|
| order_id | Associated order identifier |
| order_amount | Order amount |
| order_status | Status of the associated order |
| payment_method | Payment method used |
| tip_amount | Tip amount |
| payment_status | Payment outcome |

## 7. Reviews — `reviews.csv`

| Column | Description |
|---|---|
| review_id | Unique review identifier |
| order_id | Associated order identifier |
| customer_id | Customer who submitted the review |
| restaurant_id | Reviewed restaurant |
| rating | Customer rating |
| review_text | Written customer feedback |
| review_timestamp | Timestamp associated with the order |

## Data Notes
- All datasets contain synthetic data generated for educational purposes.
- Some missing values were intentionally introduced for data-quality analysis.
- Deliveries and reviews cover only a subset of orders.
- The data can be used for customer segmentation, sentiment analysis, delivery-time prediction, and demand forecasting.
