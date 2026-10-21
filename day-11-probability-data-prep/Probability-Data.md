# Day 11: Probability, Correlation, and Dataset Exploration

## Topics Covered

- Probability basics
- Correlation
- Train, Validation, and Test Split
- Dataset Exploration (EDA)
- Practical Data Analysis using Python

## 1. Probability

Probability measures the likelihood that an event will occur. Its value ranges from 0 to 1.

**Formula:**

Probability = Favorable Outcomes / Total Possible Outcomes

**Example:**

When rolling a fair six-sided die, the probability of getting a 3 is:

P(3) = 1 / 6 ≈ 0.167

### Python Example

```python
# Probability of getting an even number on a fair die

favorable_outcomes = 3
total_outcomes = 6

probability = favorable_outcomes / total_outcomes

print("Probability:", probability)
```

**Output:**

```text
Probability: 0.5
```

## 2. Correlation

Correlation measures the strength and direction of the relationship between two numerical variables. The Pearson correlation coefficient ranges from -1 to +1.

- **Positive correlation:** Both variables tend to increase together.
- **Negative correlation:** One variable tends to increase as the other decreases.
- **Zero correlation:** No linear relationship is detected.

### Example

Study hours and exam marks may have a positive correlation because students who study more may tend to achieve higher marks.

<LearningViz type_id="CORRELATION" initial_values={{"pattern":"positive"}}/>

### Python Example

```python
import numpy as np

study_hours = [1, 2, 3, 4, 5]
marks = [40, 50, 60, 70, 80]

correlation = np.corrcoef(study_hours, marks)[0, 1]

print("Correlation:", round(correlation, 2))
```

**Output:**

```text
Correlation: 1.0
```

Note: Correlation does not necessarily imply causation.

## 3. Train, Validation, and Test Split

Dataset splitting divides data into separate subsets to train and evaluate a machine learning model.

- **Training set:** Used to train the model.
- **Validation set:** Used to tune hyperparameters and select models.
- **Test set:** Used for final evaluation on unseen data.

### Common Split Ratios

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

### Python Example

```python
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Split into 70% training and 30% temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Split temporary data equally into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))
```

The split percentages are approximately 70%, 15%, and 15%.

**Important:** For classification, consider stratified splitting to preserve class proportions. For time-series data, split chronologically instead of randomly. Fit preprocessing steps only on the training set to prevent data leakage.

## 4. Dataset Exploration Task (EDA)

Exploratory Data Analysis (EDA) helps us understand a dataset before building a machine learning model.

### Tasks

1. Load a CSV dataset using Pandas.
2. Display the first five rows.
3. Check the number of rows and columns.
4. Identify numerical and categorical columns.
5. Check for missing values and duplicate rows.
6. Calculate descriptive statistics.
7. Explore correlations between numerical features.
8. Visualize distributions and relationships using Matplotlib and Seaborn.
9. Identify unusual values and potential outliers.
10. Write a short summary of the findings.

### Python Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Wine.csv")

# Inspect the data
print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Correlation between numerical columns
correlation = df.select_dtypes(include="number").corr()

print("\nCorrelation matrix:")
print(correlation)

# Visualize correlations
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
```

Replace `Wine.csv` with the path to your chosen CSV file. The dataset should contain numerical columns for the correlation analysis.

## Summary

| Concept | Purpose |
|---|---|
| Probability | Measures the likelihood of an event |
| Correlation | Measures the direction and strength of a linear relationship |
| Training set | Trains the model |
| Validation set | Helps select and tune the model |
| Test set | Evaluates final model performance |
| EDA | Helps understand data quality, patterns, and relationships |

## Tools and Libraries

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
