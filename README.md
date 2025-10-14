# 🛡️ Fraud Detection — A Logistic Regression Approach

A compact and reproducible pipeline that trains a Logistic Regression model to detect fraudulent transactions.This project demonstrates how to train, evaluate, and choose a threshold (probability cut-off) to classify frauds in an interpretable and scalable way, using a manually injected dataset for clear and direct learning.

---

## ✨ Key Features

- Complete Pipeline: A full demonstration of the machine learning pipeline, from feature engineering to model evaluation and inference on new data.

- Behavioral Metrics: Utilizes valuable metrics like transaction frequency and historical value to identify anomalies.

- Cyclical Time Transformation: Employs a sine-cosine transformation to ensure the model correctly interprets the cyclical nature of transaction timestamps.

- Location Comparison: Includes a feature that compares the customer's residence city with the transaction city, a strong indicator of potential fraud.

- Automatic Threshold Selection: The model automatically selects the optimal probability cut-off that maximizes the F1-Score.

- Portfolio-Ready: An ideal project for your portfolio, showcasing your skills in feature engineering and model evaluation.

---

## 📂 Repository Contents

- logisticRegression.py — The complete training and prediction script with manually injected data.
- README.md — This file.

## ⚙️ Requirements

Install dependencies using pip install:

```
pip install pandas numpy scikit-learn
```
```
pip install matplotlib
```

---

# 🧠 Feature Engineering and Analysis

- This model goes beyond basic metrics to capture complex behavioral patterns common in fraud. The following key features have been engineered and implemented:

- Cyclical Time Transformation: A standard logistic regression model doesn't understand that 11 PM and 1 AM are close in time. To solve this, we use a sine and cosine transformation on the transaction hour. This allows the model to interpret time in a cyclical manner, recognizing that transactions at the beginning and end of the day might have similar risk profiles, while late-night transactions (out of the normal pattern) are often more suspicious.

- City Comparison: One of the most critical features is the difference between the customer's resident city and the transaction city. The model is trained on a binary variable that signals if these two cities are different, which is a strong alert for purchases made away from the usual location.

- Historical Value Comparison: The model compares the current transaction value against the customer's historical average purchase value. A transaction significantly higher than the customer's usual spend is treated as an anomaly and assigned a higher fraud risk.

- Transaction Frequency: The velocity of transactions is a crucial indicator. This feature counts the number of purchases made within a short period (e.g., the last 24 hours). A high frequency of transactions in a short time frame is a classic pattern of stolen card usage.

# 🧠 Threshold Selection & Business Trade-offs

- In fraud detection problems, accuracy is not always the best metric. The choice of the probability threshold is critical for balancing precision (how many detected frauds are actually fraudulent) and recall (how many actual frauds the model successfully detects).

- Manual Definition: In this project, the threshold is manually set at 75%. This choice prioritizes precision, minimizing false positives (legitimate transactions that would be blocked). This approach is ideal when the business goal is to avoid inconveniencing the customer.

- Business Trade-offs:

- Higher Threshold (like 75%): Results in fewer fraud alerts, but may miss more actual frauds. It prioritizes customer experience.

- Lower Threshold: Generates more alerts, but catches more frauds. It prioritizes security and minimizing financial loss.

---

# ✍️ Author & Motivation
This project was developed for educational and portfolio purposes. It demonstrates a full machine learning pipeline from feature engineering and preprocessing to training, threshold selection, and inference using free and easily reproducible tools.
