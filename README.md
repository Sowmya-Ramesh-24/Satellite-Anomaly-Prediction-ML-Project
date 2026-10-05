
# Satellite Anomaly Prediction

### Survival Analysis and Machine Learning

A Stanford CS229-inspired machine learning project for predicting the **time until the next satellite anomaly** using survival analysis and regression-based machine learning techniques.

The project compares:

- Linear Regression
- Support Vector Regression (SVR)
- Modified Naive Bayes Survival Model

---

## Team 

| SOWMYA RAMESH | SRINIKESH DURGAVAJJULA |
| -------- | -------- | 
| PES2UG24CS512| PES2UG24CS520 | 



## Problem Definition & Objective

Satellite anomalies are difficult to predict because their timing depends on multiple factors, including:

- Temporal factors
- Space-weather conditions
- Orbital characteristics
- Historical anomaly patterns

### Objective

The project models **Time-to-Event (TTE)**, measured as the number of days from one known satellite anomaly to the next known anomaly.

The objective is to compare multiple machine-learning approaches for **continuous TTE prediction**.

---

## Our Approach

The project follows a complete pipeline from raw data to model comparison:

```text
Raw Data
   ↓
Data Cleaning & Alignment
   ↓
Time-to-Event (TTE) Construction
   ↓
Feature Engineering
   ↓
10-Fold Cross-Validation
   ↓
Machine Learning Models
   ↓
RAE / MAE / RMSE
   ↓
Model Comparison
