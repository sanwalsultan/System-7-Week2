# pip install pandas numpy scikit-learn


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Create a dataset
data = {
    "Study_Hours": [2, 4, 6, 8, 3, 5, 7, 9, 1, 6],
    "Attendance": [60, 70, 80, 90, 65, 75, 85, 95, 50, 80],
    "Previous_Marks": [45, 55, 65, 80, 50, 60, 75, 90, 35, 70],
    "Final_Marks": [40, 50, 65, 85, 45, 60, 80, 95, 30, 70]
}

df = pd.DataFrame(data)

# 2. Explore the dataset
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# 3. Check and remove duplicate rows
df = df.drop_duplicates()

# 4. Calculate statistics using NumPy
marks = df["Final_Marks"].to_numpy()

print("\nMean:", np.mean(marks))
print("Median:", np.median(marks))
print("Standard deviation:", np.std(marks))

# 5. Prepare features (X) and target (y)
X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
