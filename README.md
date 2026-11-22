# Olist E-Commerce Delivery Delay Prediction

A machine learning project that predicts whether an e-commerce order will be delivered late (`is_late = 1`) or on time (`0`) using the Brazilian Olist E-Commerce dataset, wrapped in a lightweight Streamlit prototype.

## Project Overview

Late deliveries directly impact customer satisfaction and operational planning. This project frames delivery delay prediction as a binary classification problem under class imbalance, moving from preprocessing and feature engineering to model evaluation, threshold tuning, and local deployment.

## Key Features & Pipeline

- **Preprocessing:** Built an end-to-end scikit-learn `Pipeline` with `ColumnTransformer` to handle numerical and categorical features cleanly without data leakage.
- **Engineered Features:**
  - `approval_hours` — time taken from order purchase to payment approval
  - `estimated_delivery_days` — promised delivery window
  - `same_state` — binary indicator for whether buyer and seller are in the same state
  - `freight_to_price_ratio` — shipping cost relative to item price
  - `product_volume_cm3` — derived product dimensions
  - Temporal signals (`purchase_dayofweek`, `purchase_month`)
- **Model:** `HistGradientBoostingClassifier` exported via `joblib` (`olist_delay_hgb_model.joblib`).

## Evaluation & Insights

- **ROC-AUC:** `0.8034`
- **F1-Score:** `~0.32` (reflecting the natural class imbalance in late vs. on-time orders)
- **Threshold Tuning:** Evaluated decision thresholds beyond the default `0.50`, finding `0.65` achieves the highest F1-score and illustrates the practical trade-off between catching late orders (recall) and avoiding false alarms (precision).
- **Feature Importance:** Permutation importance analysis highlighted estimated delivery window, freight-to-price ratio, and location/state factors as key drivers.

## Repository Structure

```text
├── app.py                        # Streamlit web application
├── olist_delay_hgb_model.joblib  # Trained scikit-learn pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
