# System-7 Week 2 — Machine Learning Study Plan

Notes and hands-on examples for a 7-day machine learning study block (Days 8–14). Each day has its own Markdown file with concept explanations, formulas, and runnable Python examples using NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

## Weekly Plan

| Day | Topics | Notes |
|---|---|---|
| Day 8 | What is AI, ML and Deep Learning; supervised, unsupervised, reinforcement learning; dataset terms | [Ml-Foundation.md](day-08-ml-foundations/Ml-Foundation.md) |
| Day 9 | Mean, median, mode, variance, standard deviation | [Descriptive-Statistics.md](day-09-descriptive-statistics/Descriptive-Statistics.md) |
| Day 10 | Linear algebra basics: vectors, matrices, dot products, matrix multiplication | [Linear-Algebra.md](day-10-linear-algebra/Linear-Algebra.md) |
| Day 11 | Probability, correlation, train/validation/test split; EDA on `Wine.csv` | [Probability-Data.md](day-11-probability-data-prep/Probability-Data.md) |
| Day 12 | Linear and multiple linear regression; classification vs. regression; Scikit-learn workflow | [Regression-Sklearn.md](day-12-regression-sklearn/Regression-Sklearn.md) |
| Day 13 | Logistic regression, KNN, decision trees, random forests; classification metrics | [Classification-Models.md](day-13-classification-models/Classification-Models.md) |
| Day 14 | Predictions, coefficients, feature importance; house-price and classification tasks | [House-Price.md](day-14-house-price-classifier/House-Price.md) |

## Progress

- [x] Day 8 — AI, ML and Deep Learning; learning types; dataset terms → [notes](day-08-ml-foundations/Ml-Foundation.md)
- [x] Day 9 — Mean, median, mode, variance, standard deviation → [notes](day-09-descriptive-statistics/Descriptive-Statistics.md)
- [x] Day 10 — Vectors, matrices, dot products → [notes](day-10-linear-algebra/Linear-Algebra.md)
- [x] Day 11 — Probability, correlation, data splits, EDA → [notes](day-11-probability-data-prep/Probability-Data.md)
- [x] Day 12 — Linear and multiple linear regression; Scikit-learn workflow → [notes](day-12-regression-sklearn/Regression-Sklearn.md)
- [x] Day 13 — Logistic regression, KNN, decision trees, random forests → [notes](day-13-classification-models/Classification-Models.md)
- [x] Day 14 — Predictions, coefficients, feature importance → [notes](day-14-house-price-classifier/House-Price.md)

## Repository Structure

```text
.
├── day-08-ml-foundations/
│   └── Ml-Foundation.md
├── day-09-descriptive-statistics/
│   └── Descriptive-Statistics.md
├── day-10-linear-algebra/
│   └── Linear-Algebra.md
├── day-11-probability-data-prep/
│   ├── Probability-Data.md
│   └── Wine.csv                              # Sample dataset for the EDA task
├── day-12-regression-sklearn/
│   └── Regression-Sklearn.md
├── day-13-classification-models/
│   └── Classification-Models.md
└── day-14-house-price-classifier/
    └── House-Price.md                        # Day 14 notes
```

## What Each Day Covers

- **Day 8 — ML Foundations:** AI vs. ML vs. DL, supervised/unsupervised/reinforcement learning, dataset terminology (features, labels, splits, preprocessing, class imbalance), key ML terms (parameters, hyperparameters, overfitting, data leakage), evaluation metrics, and the standard ML workflow.
- **Day 9 — Descriptive Statistics:** Mean, median, mode, variance, and standard deviation, with NumPy examples and a complete summary program.
- **Day 10 — Linear Algebra:** Vectors, matrices, vector addition, matrix addition, dot products, and matrix multiplication using NumPy.
- **Day 11 — Probability & Data Prep:** Probability basics, correlation, 70/15/15 train-validation-test splitting, and an end-to-end EDA task (inspect, summarize, correlate, visualize) on `Wine.csv`.
- **Day 12 — Regression with Scikit-learn:** Simple and multiple linear regression, regression vs. classification, the `fit`/`predict`/evaluate workflow, and MAE, MSE, RMSE, and R² metrics.
- **Day 13 — Classification Models:** Logistic regression, KNN, decision trees, and random forests, plus pipelines and evaluation with accuracy, precision, recall, F1-score, and a confusion matrix.
- **Day 14 — Predictions & Interpretation:** Using `predict()`, reading `coef_`/`intercept_`, tree-based `feature_importances_`, and full regression (house price) and classification (breast cancer) examples.

## Tools and Libraries

The code examples use:

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Install the dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run any example by copying the code block into a Python script or Jupyter notebook. For Day 11's EDA example, run the script from `day-11-probability-data-prep/` so that `pd.read_csv("Wine.csv")` resolves correctly.
