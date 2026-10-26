# Data Quality Report

## 1. Overview
Seven synthetic datasets were checked for row counts, column counts, missing values, and fully duplicated rows.

## 2. Results

| Dataset | Rows | Columns | Missing Values | Duplicate Rows |
|---|---:|---:|---:|---:|
| Customers | 1,000 | 5 | 10 in age | 0 |
| Restaurants | 100 | 6 | 1 in restaurant_rating | 0 |
| Drivers | 200 | 5 | 0 | 0 |
| Orders | 10,000 | 9 | 0 | 0 |
| Deliveries | 8,831 | 6 | 0 | 0 |
| Payments | 10,000 | 6 | 0 | 0 |
| Reviews | 5,740 | 7 | 57 in review_text | 0 |

## 3. Observations
- No fully duplicated rows were detected.
- Missing values were found in customer age, restaurant rating, and review text.
- The missing values were intentionally introduced during synthetic data generation.
- Missing review text should be handled before text analysis.
- Missing ages and restaurant ratings should be considered during feature engineering and analysis.

## 4. Limitations
The initial check does not verify unique identifiers, valid value ranges, or relationships between datasets. These checks should be performed during subsequent data preparation.

## 5. Conclusion
The datasets are ready for further exploratory checks, with missing values documented for later handling.
