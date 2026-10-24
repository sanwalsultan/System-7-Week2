
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

DATA_PATH = Path(__file__).parent / "House Price Prediction Dataset.csv"
OUTPUT_FOLDER = Path(__file__).parent / "house_price_plots"
OUTPUT_FOLDER.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("First five houses:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. CLEAN THE DATASET
# ============================================================

df = df.drop_duplicates().copy()

numeric_columns = [
    "Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"
]
categorical_columns = ["Location", "Condition", "Garage"]

# Convert numeric columns to numeric types.
# Invalid values become NaN and will be handled below.
for column in numeric_columns + ["Price"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Price is the target, so remove rows without a valid price.
df = df.dropna(subset=["Price"]).copy()

# Fill missing numeric features using their medians.
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing categorical features using their most common values.
for column in categorical_columns:
    mode_values = df[column].mode()
    fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
    df[column] = df[column].fillna(fill_value)

print("\nCleaned dataset shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 3. GRAPH 1: AREA VS PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="Area",
    y="Price",
    scatter_kws={"alpha": 0.5},
    line_kws={"linewidth": 2},
    ci=None
)

plt.title("House Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "01_area_vs_price.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 4. GRAPH 2: NUMERIC FEATURES VS PRICE
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for index, column in enumerate(numeric_columns):
    sns.scatterplot(
        data=df,
        x=column,
        y="Price",
        alpha=0.5,
        ax=axes[index]
    )
    axes[index].set_title(f"{column} vs Price")
    axes[index].set_xlabel(column)
    axes[index].set_ylabel("Price")

# Hide the unused sixth subplot.
axes[5].set_visible(False)

fig.suptitle("How Numeric Features Relate to House Prices", fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "02_numeric_features_vs_price.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 5. GRAPH 3: PRICE BY LOCATION, CONDITION, AND GARAGE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

for index, column in enumerate(categorical_columns):
    sns.boxplot(
        data=df,
        x=column,
        y="Price",
        ax=axes[index]
    )

    axes[index].set_title(f"Price by {column}")
    axes[index].set_xlabel(column)
    axes[index].set_ylabel("Price")
    axes[index].tick_params(axis="x", rotation=25)

fig.suptitle("How Categorical Features Relate to House Prices", fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "03_categorical_features_vs_price.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 6. GRAPH 4: AVERAGE PRICE BY CATEGORY
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for index, column in enumerate(categorical_columns):
    average_prices = (
        df.groupby(column, observed=True)["Price"]
        .mean()
        .sort_values(ascending=False)
    )

    sns.barplot(
        x=average_prices.index,
        y=average_prices.values,
        ax=axes[index]
    )

    axes[index].set_title(f"Average Price by {column}")
    axes[index].set_xlabel(column)
    axes[index].set_ylabel("Average Price")
    axes[index].tick_params(axis="x", rotation=25)

fig.suptitle("Which Categories Have Higher Average Prices?", fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "04_average_price_by_category.png", dpi=150)
plt.show()
plt.close()


# Print average and median prices for comparison.
for column in categorical_columns:
    print(f"\nPrice statistics by {column}:")
    print(
        df.groupby(column, observed=True)["Price"]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
        .round(2)
    )


# ============================================================
# 7. GRAPH 5: CORRELATION HEATMAP
# ============================================================

correlation_columns = numeric_columns + ["Price"]
correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Numeric Features and Price")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "05_correlation_heatmap.png", dpi=150)
plt.show()
plt.close()

print("\nCorrelation with Price:")
print(
    correlation_matrix["Price"]
    .drop("Price")
    .sort_values(ascending=False)
)


# ============================================================
# 8. SIMPLE LINEAR REGRESSION
# Use only Area to predict Price.
# ============================================================

X_simple = df[["Area"]]
y = df["Price"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple,
    y,
    test_size=0.2,
    random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

simple_predictions = simple_model.predict(X_test_s)

print("\n===== SIMPLE LINEAR REGRESSION =====")
print("Feature: Area")
print("Coefficient:", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)
print("MAE:", mean_absolute_error(y_test_s, simple_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test_s, simple_predictions)))
print("R2 score:", r2_score(y_test_s, simple_predictions))

# Test on unseen data using Area only
print("\n===== SIMPLE LINEAR REGRESSION: UNSEEN DATA =====")

unseen_area = 3500

unseen_house = pd.DataFrame({
    "Area": [unseen_area]
})

predicted_price_simple = simple_model.predict(unseen_house)[0]

print(f"House Area: {unseen_area:,}")
print(f"Predicted House Price: {predicted_price_simple:,.2f}")

# ============================================================
# 9. MULTIPLE LINEAR REGRESSION
# Use numeric and categorical features.
# ============================================================

X = df[numeric_columns + categorical_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert categorical values into numeric columns.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

multiple_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])

multiple_model.fit(X_train, y_train)

predictions = multiple_model.predict(X_test)

print("\n===== MULTIPLE LINEAR REGRESSION =====")
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2 score:", r2_score(y_test, predictions))


# ============================================================
# 10. GRAPH 6: ACTUAL PRICE VS PREDICTED PRICE
# ============================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test,
    y=predictions,
    alpha=0.6
)

# Reference line: perfect predictions would fall on this line.
minimum_price = min(y_test.min(), predictions.min())
maximum_price = max(y_test.max(), predictions.max())

plt.plot(
    [minimum_price, maximum_price],
    [minimum_price, maximum_price],
    linestyle="--"
)

plt.title("Actual Prices vs Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "06_actual_vs_predicted.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 11. DISPLAY AND PLOT LEARNED COEFFICIENTS
# ============================================================

feature_names = multiple_model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = multiple_model.named_steps["regression"].coef_

coefficient_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coefficient_table["Absolute_Coefficient"] = (
    coefficient_table["Coefficient"].abs()
)

coefficient_table = coefficient_table.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\nLearned coefficients:")
print(
    coefficient_table[
        ["Feature", "Coefficient"]
    ].to_string(index=False)
)

print(
    "\nIntercept:",
    multiple_model.named_steps["regression"].intercept_
)

# Plot the features with the largest absolute coefficients.
top_coefficients = (
    coefficient_table.head(15)
    .sort_values("Coefficient")
)

plt.figure(figsize=(11, 7))

sns.barplot(
    data=top_coefficients,
    x="Coefficient",
    y="Feature"
)

plt.axvline(0, linestyle="--")
plt.title("Top 15 Regression Coefficients")
plt.xlabel("Coefficient (positive or negative)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "07_regression_coefficients.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 12. PREDICT THE PRICE OF A NEW HOUSE
# ============================================================

new_house = pd.DataFrame([{
    "Area": 2500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Floors": 2,
    "YearBuilt": 2010,
    "Location": "Suburban",
    "Condition": "Good",
    "Garage": "Yes"
}])

predicted_price = multiple_model.predict(new_house)[0]

print("\n===== NEW HOUSE PREDICTION =====")
print("House details:")
print(new_house.to_string(index=False))
print(f"Predicted price: {predicted_price:,.2f}")


# ============================================================
# 13. FINAL INTERPRETATION
# ============================================================

r2 = r2_score(y_test, predictions)

print("\n===== FINAL INTERPRETATION =====")

if r2 > 0:
    print(f"The model explains approximately {r2 * 100:.2f}% of price variation.")
else:
    print(
        "The model does not explain price variation well on this test set. "
        "Its R2 score is zero or negative."
    )

print("\nAll graphs saved in:")
print(OUTPUT_FOLDER.resolve())


# Limitations using Linear Regression

'''Inability to Capture Complex Patterns: Linear regression is restricted to finding a single straight line (or flat plane) using linear weights.
Because real-world or synthetic housing patterns often contain non-linear relationships, multi-variable interactions, or random noise, a rigid linear approach completely misses them.
Near-Zero Correlation and Flat Predictions:Since the dataset features have virtually no statistical correlation with price, the model defaults to learning a near-zero slope, resulting in an $R^2$ score close to zero or negative and a flat cluster of predictions.'''


