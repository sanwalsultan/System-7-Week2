# Day 12: Linear Regression and Scikit-learn Workflow

## Topics Covered

- Linear Regression
- Multiple Linear Regression
- Classification vs. Regression
- Scikit-learn Workflow
- Model Training and Evaluation

## 1. Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict a **continuous numerical value** based on one input feature.

**Example:** Predicting a student's marks based on study hours.

### Formula

Prediction = Intercept + (Coefficient × Input)

### Python Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Input feature: study hours
X = np.array([[1], [2], [3], [4], [5]])

# Target: student marks
y = np.array([40, 50, 60, 70, 80])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for 6 study hours
prediction = model.predict([[6]])

print("Predicted Marks:", prediction[0])
```

**Expected Output:**

```text
Predicted Marks: 90.0
```

### Explanation

- `X` contains the input feature (study hours).
- `y` contains the target values (student marks).
- `LinearRegression()` creates the model.
- `fit()` trains the model.
- `predict()` predicts marks for a new input.

## 2. Multiple Linear Regression

Multiple Linear Regression predicts a continuous value using **two or more input features**.

**Example:** Predicting house prices using house size, number of bedrooms, and house age.

### Formula

Prediction = Intercept + (Coefficient₁ × Feature₁) + (Coefficient₂ × Feature₂) + ...

### Python Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Features: house size, bedrooms, age
X = np.array([
    [1000, 2, 10],
    [1500, 3, 8],
    [2000, 3, 5],
    [2500, 4, 3],
    [3000, 4, 1]
])

# Target: house price in thousands
y = np.array([100, 150, 200, 250, 300])

model = LinearRegression()
model.fit(X, y)

# Predict for a house with size 2200, 3 bedrooms, age 4
prediction = model.predict([[2200, 3, 4]])

print("Predicted House Price:", prediction[0])
```

**Note:** This is a small illustrative dataset. Real-world predictions require representative data and proper model evaluation.

## 3. Classification vs. Regression

Both are supervised machine learning tasks, but they predict different types of outputs.

| Feature | Classification | Regression |
|---|---|---|
| Purpose | Predict a category | Predict a numerical value |
| Output | Discrete class or label | Continuous number |
| Example | Spam or Not Spam | House price |
| Algorithms | Logistic Regression, Decision Tree Classifier | Linear Regression, Random Forest Regressor |
| Evaluation | Accuracy, Precision, Recall, F1-score | MAE, MSE, RMSE, R² |

### Examples

**Classification:**
- Predict whether an email is spam.
- Predict whether a transaction is fraudulent.
- Predict whether a customer will leave.

**Regression:**
- Predict house prices.
- Predict monthly sales.
- Predict temperature.

**Important:** Despite its name, Logistic Regression is commonly used for classification.

## 4. Scikit-learn Workflow

Scikit-learn provides tools to prepare data, train machine learning models, make predictions, and evaluate performance.

### Standard Workflow

1. Define the problem.
2. Load the dataset.
3. Explore and clean the data.
4. Select features (`X`) and target (`y`).
5. Split the dataset into training and test sets.
6. Create a model.
7. Train the model using `fit()`.
8. Generate predictions using `predict()`.
9. Evaluate the model.
10. Save the trained model if required.

### Example: Complete Workflow

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load sample data
X = np.array([[1], [2], [3], [4], [5],
              [6], [7], [8], [9], [10]])

y = np.array([20, 30, 40, 50, 60,
              70, 80, 90, 100, 110])

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create the model
model = LinearRegression()

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict on unseen test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 2))

# 7. Predict a new value
new_prediction = model.predict([[11]])
print("Prediction for input 11:", new_prediction[0])
```

### Understanding Important Functions

| Function | Purpose |
|---|---|
| `train_test_split()` | Divides data into training and test sets |
| `LinearRegression()` | Creates a regression model |
| `fit()` | Trains the model |
| `predict()` | Generates predictions |
| `mean_absolute_error()` | Measures average absolute prediction error |
| `r2_score()` | Measures how well the model explains target variation |

## 5. Model Evaluation Metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values. Lower values generally indicate better predictions.

### Mean Squared Error (MSE)

MSE calculates the average squared prediction error. Large errors receive greater penalties.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and uses the same units as the target variable.

### R² Score

R² measures how well the model explains variation in the target. A score closer to 1 generally indicates a better fit, but it should be evaluated on unseen data.

## Summary

| Concept | Purpose |
|---|---|
| Linear Regression | Predicts numerical values using one feature |
| Multiple Linear Regression | Predicts numerical values using multiple features |
| Classification | Predicts categories |
| Regression | Predicts numerical values |
| Scikit-learn | Provides tools for machine learning |
| `fit()` | Trains a model |
| `predict()` | Generates predictions |
| MAE and RMSE | Measure prediction errors |
| R² | Measures explained variation |


## Tools and Libraries

- Python
- NumPy
- Scikit-learn
