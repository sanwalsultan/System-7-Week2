# Day 14 — Model Predictions, Coefficients, and Feature Importance

## Overview

Day 14 focuses on understanding model predictions, interpreting coefficients, and identifying important features in machine learning. We will practice both regression and classification tasks using Scikit-learn.

## Topics Covered

- Making predictions with trained models
- Understanding model coefficients
- Understanding feature importance
- House-price prediction using regression
- Classification using Logistic Regression and Random Forest
- Evaluating regression and classification models

## 1. Making Predictions

After training a machine learning model, we use the `predict()` method to make predictions on new data.

```python
from sklearn.linear_model import LinearRegression

# Input feature: house size in square feet
X = [[800], [1000], [1200], [1500], [1800]]

# Target: house price in thousands
y = [80, 100, 120, 150, 180]

model = LinearRegression()
model.fit(X, y)

# Predict the price of a 1300 sq ft house
prediction = model.predict([[1300]])

print("Predicted house price:", prediction[0], "thousand")
```

**Key points:**
- `fit(X, y)` trains the model.
- `predict()` generates predictions for new inputs.
- The input features used for prediction must match the features used during training.

## 2. Understanding Model Coefficients

Coefficients indicate how an input feature relates to the prediction in a linear model.

For simple linear regression:

\[
y=b_0+b_1x
\]

- \(b_0\): intercept, or predicted output when the input is zero.
- \(b_1\): coefficient, representing the change in predicted output for a one-unit increase in the input.

### Python example

```python
from sklearn.linear_model import LinearRegression

X = [[800], [1000], [1200], [1500], [1800]]
y = [80, 100, 120, 150, 180]

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
```

In this example, the coefficient represents the predicted increase in house price, in thousands, for each additional square foot.

**Important:** Coefficients describe the model's learned relationship, not necessarily a causal relationship. Their magnitudes also depend on feature units and scaling.

## 3. Multiple Linear Regression Coefficients

Multiple linear regression uses two or more input features.

Example: Predict house price using house size and number of bedrooms.

```python
from sklearn.linear_model import LinearRegression

# Features: size in square feet, number of bedrooms
X = [
    [800, 2],
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4]
]

# House prices in thousands
y = [80, 100, 125, 155, 190]

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

prediction = model.predict([[1300, 3]])
print("Predicted price:", prediction[0], "thousand")
```

Each coefficient corresponds to a feature in the same order as the columns in `X`.

## 4. Feature Importance

Feature importance helps identify which input features a model relies on most when making predictions.

For example, in house-price prediction, useful features might include:
- House size
- Number of bedrooms
- Location
- Property age

A Random Forest can provide a feature-importance score using `feature_importances_`.

```python
from sklearn.ensemble import RandomForestRegressor

X = [
    [800, 2],
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2000, 4]
]

# Features: size, bedrooms
y = [80, 100, 125, 155, 190, 210]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

feature_names = ["House Size", "Bedrooms"]

for name, importance in zip(
    feature_names,
    model.feature_importances_
):
    print(name, ":", round(importance, 3))
```

**Key points:**
- Higher importance means a feature contributed more to the forest's impurity-based importance calculation.
- Importance scores are not the same as regression coefficients.
- Importance does not prove causation.
- Correlated features can share or distort importance, so interpret results carefully.

## 5. House-Price Prediction Task (Regression)

The goal is to predict a continuous numerical value: house price.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

X = [
    [800], [1000], [1200], [1400],
    [1600], [1800], [2000], [2200]
]

y = [80, 100, 120, 140, 160, 180, 200, 220]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual prices:", y_test)
print("Predicted prices:", y_pred)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

### Regression metrics

- **MAE:** Average absolute difference between actual and predicted values.
- **R²:** Measures how well the model explains variation in the target relative to a mean-prediction baseline. It can be negative on test data.

## 6. Classification Task

Classification predicts a class, such as spam/not spam or pass/fail.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Inspect classification feature importance

```python
for name, importance in zip(
    data.feature_names,
    model.feature_importances_
):
    print(name, ":", round(importance, 4))
```

This prints the importance score for each feature used by the Random Forest.

## 7. Regression vs. Classification

| Aspect | Regression | Classification |
|---|---|---|
| Prediction | Continuous numerical value | Category or class |
| Example | House price | Spam or not spam |
| Common models | Linear Regression, Random Forest Regressor | Logistic Regression, Random Forest Classifier |
| Common metrics | MAE, RMSE, R² | Accuracy, precision, recall, F1-score |

## 8. Typical Machine Learning Workflow

1. Define the prediction problem.
2. Load and explore the dataset.
3. Clean missing or invalid values.
4. Separate features (`X`) and target (`y`).
5. Split the data into training and test sets.
6. Train the model on the training data.
7. Generate predictions on unseen test data.
8. Evaluate predictions using appropriate metrics.
9. Inspect coefficients or feature importance when supported by the model.
10. Compare models and improve them using validation data or cross-validation.

## Summary

| Concept | Meaning |
|---|---|
| `predict()` | Generates predictions from a trained model |
| `coef_` | Coefficients of a fitted linear model |
| `intercept_` | Intercept of a fitted linear model |
| `feature_importances_` | Tree-based feature-importance scores |
| Regression | Predicts numerical values |
| Classification | Predicts categories |


## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

