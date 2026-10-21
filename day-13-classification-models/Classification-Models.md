# Day 13 — Classification Algorithms in Machine Learning

## Overview

Day 13 focuses on classification algorithms used to predict categories or classes from data. We will learn Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forests using Python and Scikit-learn.

## Topics Covered

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Training and testing classification models
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

## 1. What Is Classification?

Classification is a supervised machine learning task where a model predicts a **category or class**.

Examples:
- Email: Spam or Not Spam
- Customer: Will Buy or Will Not Buy
- Transaction: Fraud or Not Fraud
- Student: Pass or Fail

Unlike regression, which predicts continuous numerical values, classification predicts discrete classes.

## 2. Logistic Regression

Logistic Regression is a classification algorithm commonly used for binary classification. It estimates the probability that an input belongs to a class.

**Example:** Predict whether a student passes or fails based on study hours.

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]  # 0 = Fail, 1 = Pass

model = LogisticRegression()
model.fit(X, y)

prediction = model.predict([[4]])

print("Predicted class:", prediction[0])
print("Class probabilities:", model.predict_proba([[4]]))
```

**Key points:**
- Used for classification, not ordinary continuous-value regression.
- Can estimate class probabilities.
- Commonly used for binary classification and can also support multiclass classification.

## 3. K-Nearest Neighbors (KNN)

KNN predicts a class by looking at the nearest training examples. The majority class among the nearest neighbors determines the prediction.

**Example:** Classify a new student based on similar students.

```python
from sklearn.neighbors import KNeighborsClassifier

X = [[1], [2], [3], [6], [7], [8]]
y = [0, 0, 0, 1, 1, 1]  # 0 = Fail, 1 = Pass

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

prediction = model.predict([[5]])

print("Predicted class:", prediction[0])
```

**Key points:**
- `n_neighbors=3` means the model considers the three nearest training examples.
- Feature scaling is important because KNN uses distances.
- Predictions can become slower when the training dataset is large.

## 4. Decision Tree Classifier

A Decision Tree learns rules from data and uses them to predict a class.

**Example:** Predict whether a student passes based on study hours and attendance.

```python
from sklearn.tree import DecisionTreeClassifier

X = [
    [1, 50],
    [2, 60],
    [5, 80],
    [6, 90],
    [3, 55],
    [7, 95]
]

y = [0, 0, 1, 1, 0, 1]  # 0 = Fail, 1 = Pass

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

prediction = model.predict([[5, 75]])

print("Predicted class:", prediction[0])
```

**Key points:**
- Learns decision rules from features.
- Can model non-linear decision boundaries.
- Easy to visualize and interpret.
- Deep trees may overfit the training data.

## 5. Random Forest Classifier

Random Forest combines multiple decision trees. Each tree makes a prediction, and the forest combines their votes to determine the final class.

**Example:** Predict whether a transaction is fraudulent using transaction features.

```python
from sklearn.ensemble import RandomForestClassifier

X = [
    [20, 0],
    [50, 0],
    [500, 1],
    [800, 1],
    [30, 0],
    [700, 1]
]

y = [0, 0, 1, 1, 0, 1]  # 0 = Not Fraud, 1 = Fraud

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

prediction = model.predict([[600, 1]])

print("Predicted class:", prediction[0])
```

**Key points:**
- `n_estimators=100` creates a forest of 100 trees.
- Often generalizes better than a single decision tree.
- Can learn complex patterns and feature interactions.
- Usually requires more computation than a single tree.

## 6. Compare the Algorithms

| Algorithm | Main idea | Feature scaling | Main consideration |
|---|---|---|---|
| Logistic Regression | Estimates class probabilities | Often helpful | Learns a linear decision boundary by default |
| KNN | Uses nearest examples | Important | Prediction can be slow on large datasets |
| Decision Tree | Learns decision rules | Usually unnecessary | Can overfit |
| Random Forest | Combines many decision trees | Usually unnecessary | More computationally expensive |

## 7. Train and Evaluate a Classification Model

The following example uses Logistic Regression and evaluates its predictions on a held-out test set.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load a sample classification dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create a pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict on unseen test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### Understanding the evaluation metrics

- **Accuracy:** Percentage of predictions that are correct.
- **Precision:** Of the examples predicted as positive, how many are actually positive?
- **Recall:** Of all actual positive examples, how many did the model identify?
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Shows correct and incorrect predictions by class.

Accuracy alone can be misleading when classes are imbalanced, so examine precision, recall, and F1-score as well.

## 8. Typical Machine Learning Workflow

1. Define the classification problem.
2. Load the dataset.
3. Explore and clean the data.
4. Separate input features (`X`) and target labels (`y`).
5. Split data into training and test sets.
6. Preprocess the features, including scaling when appropriate.
7. Train the classification model.
8. Make predictions on test data.
9. Evaluate the model using suitable metrics.
10. Compare models and tune hyperparameters using validation data or cross-validation.

## Summary

| Concept | Description |
|---|---|
| Classification | Predicts categories or class labels |
| Logistic Regression | Probability-based classification |
| KNN | Classifies using nearby training examples |
| Decision Tree | Classifies using learned rules |
| Random Forest | Combines multiple decision trees |
| Scikit-learn | Python library for building machine learning models |


## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
