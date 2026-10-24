
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = Path(__file__).parent / "heart.csv"
OUTPUT_FOLDER = Path(__file__).parent / "heart_classification_plots"
OUTPUT_FOLDER.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. DEFINE TARGET LABELS
# ============================================================

# This project assumes:
# Target 0 = Has heart disease
# Target 1 = Does not have heart disease

TARGET_LABELS = {
    0: "Has heart disease",
    1: "Does not have heart disease"
}

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

print("\nTarget meaning:")
for label, meaning in TARGET_LABELS.items():
    print(f"Target {label} = {meaning}")


# ============================================================
# 3. CLEAN DATASET
# ============================================================

df = df.drop_duplicates().copy()

# Convert feature columns to numeric values.
feature_columns = [column for column in df.columns if column != "target"]

for column in feature_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["target"] = pd.to_numeric(df["target"], errors="coerce")

# Remove rows without valid target labels.
df = df.dropna(subset=["target"]).copy()

# Keep only the two expected target classes.
df = df[df["target"].isin([0, 1])].copy()
df["target"] = df["target"].astype(int)

# Fill missing feature values with their medians.
for column in feature_columns:
    median_value = df[column].median()

    if pd.isna(median_value):
        df = df.drop(columns=[column])
        feature_columns.remove(column)
        continue

    df[column] = df[column].fillna(median_value)

print("\nCleaned dataset shape:", df.shape)

X = df[feature_columns]
y = df["target"]


# ============================================================
# 4. EXPLORE TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="target",
    order=[0, 1]
)

plt.title("Heart Disease Class Distribution")
plt.xlabel("Target (0 = Disease, 1 = No Disease)")
plt.ylabel("Number of Records")
plt.xticks(
    [0, 1],
    ["0: Has Disease", "1: No Disease"]
)

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "01_target_distribution.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 5. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 9))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Heart Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "02_correlation_heatmap.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 6. COMPARE AVERAGE FEATURES BY TARGET
# ============================================================

print("\n===== AVERAGE FEATURES BY TARGET =====")

print(
    df.groupby("target")[feature_columns]
    .mean()
    .round(2)
)

# Plot selected features for comparison.
selected_features = [
    column for column in
    ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    if column in df.columns
]

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 9)
)

axes = axes.flatten()

for index, column in enumerate(selected_features):
    sns.boxplot(
        data=df,
        x="target",
        y=column,
        order=[0, 1],
        ax=axes[index]
    )

    axes[index].set_title(f"{column} by Target")
    axes[index].set_xlabel("0 = Disease, 1 = No Disease")

for index in range(len(selected_features), len(axes)):
    axes[index].set_visible(False)

plt.suptitle("Feature Distributions by Target Class", fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "03_features_by_target.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 7. SPLIT TRAINING AND TEST DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 8. CREATE CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
}


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Has heart disease",
                "Does not have heart disease"
            ],
            zero_division=0
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=[
            "Has Disease (0)",
            "No Disease (1)"
        ],
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        OUTPUT_FOLDER / f"confusion_matrix_{filename}.png",
        dpi=150
    )

    plt.show()
    plt.close()


# ============================================================
# 10. COMPARE MODEL ACCURACY
# ============================================================

results_df = pd.DataFrame(results).sort_values(
    "Accuracy",
    ascending=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.ylim(0, 1)
plt.title("Classification Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "04_model_comparison.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 11. DECISION TREE FEATURE IMPORTANCE
# ============================================================

tree_model = trained_models["Decision Tree"]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\n===== DECISION TREE FEATURE IMPORTANCE =====")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Decision Tree Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "05_feature_importance.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 12. VISUALIZE DECISION TREE
# ============================================================

plt.figure(figsize=(22, 12))

plot_tree(
    tree_model,
    feature_names=list(X.columns),
    class_names=[
        "Has Disease (0)",
        "No Disease (1)"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Heart Classification")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "06_decision_tree.png", dpi=150)
plt.show()
plt.close()


# ============================================================
# 13. PREDICT AN UNSEEN EXAMPLE
# ============================================================

new_person = pd.DataFrame([{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
}])

# Match the feature columns and their order to the training dataset.
new_person = new_person.reindex(columns=X.columns)

print("\n===== UNSEEN EXAMPLE PREDICTIONS =====")
print(new_person.to_string(index=False))

for name, model in trained_models.items():

    prediction = int(model.predict(new_person)[0])

    print(f"\nModel: {name}")
    print("Predicted target:", prediction)
    print("Predicted class:", TARGET_LABELS[prediction])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(new_person)[0]

        print("Class probabilities:")

        for label, probability in zip(
            model.classes_,
            probabilities
        ):
            print(
                f"  {TARGET_LABELS[int(label)]}: "
                f"{probability:.2%}"
            )


# ============================================================
# 14. FINISH
# ============================================================

print("\nAll graphs saved in:")
print(OUTPUT_FOLDER.resolve())

