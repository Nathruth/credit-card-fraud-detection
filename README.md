# credit-card-fraud-detection


## 📖 Problem Description

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent a very small fraction of all observations. The goal of this project is to identify fraudulent transactions while minimizing false positives and false negatives.

This project was developed as part of ML Zoomcamp – Capstone 1, with a focus on:

* Model evaluation under class imbalance
* Comparison of classical ML models
* Deployment-oriented neural network inference using ONNX

---

## 🧠 Solution Approach

The solution follows a multi-stage approach:

1. Exploratory Data Analysis (EDA) to understand class imbalance, transaction behavior, and temporal patterns
2. Baseline classical models trained locally (Logistic Regression, XGBoost)
3. Performance comparison using ROC-AUC and classification metrics
4. Neural network model trained in Google Colab and exported to ONNX for lightweight deployment

Classical models are used as strong baselines, while the neural model is included to demonstrate deep learning deployment practices.

---

## 📊 Exploratory Data Analysis

Key EDA findings include:

* Severe class imbalance with fraud transactions representing less than 1% of the data
* Fraud rate variation by hour of day, indicating temporal patterns
* Distinct transaction amount distributions for fraud and non-fraud cases
* Feature correlations among anonymized PCA components

The following visualizations were created:

* Class imbalance distribution
* Transaction amount distribution (log scale)
* Fraud rate by hour of day
* Feature correlation heatmap

Additional feature distributions by class are provided in the Jupyter notebook.

---

## 🧪 Modeling & Evaluation

### Baseline Models

* Logistic Regression (with class weighting)
* XGBoost (with imbalance-aware weighting)

### Evaluation Metrics

Due to extreme class imbalance, models were evaluated using:
* ROC-AUC
* Precision
* Recall
* F1-score

Regression metrics such as RMSE were intentionally avoided.

---

## 📈 Model Comparison

| Model               | ROC-AUC | Precision (Fraud) | Recall (Fraud) | F1-score |
| ------------------- | ------- | ----------------- | -------------- | -------- |
| Logistic Regression | 0.968   | 0.06              | 0.87           | 0.10     |
| XGBoost             | 0.982   | 0.87              | 0.78           | 0.82     |

The ROC curve comparison shows that XGBoost provides superior discrimination and a significantly better precision–recall balance.

---

## 🔍 Interpretation

While Logistic Regression achieves a high ROC-AUC score, it produces an excessive number of false positives, making it impractical for real-world fraud detection.

XGBoost demonstrates a more balanced trade-off between precision and recall, making it a more suitable baseline model.

ROC-AUC alone is insufficient for evaluating fraud detection systems; precision and recall must be considered due to the highly imbalanced nature of the dataset.

