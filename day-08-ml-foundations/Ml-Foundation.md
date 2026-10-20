# ML Foundations — AI, Machine Learning, and Deep Learning

## 1. Introduction

Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL) are technologies used to build systems that can learn, recognize patterns, make predictions, and solve problems.

This repository covers fundamental AI and ML concepts, learning approaches, dataset terminology, model training, and evaluation.

## 2. What Is Artificial Intelligence (AI)?

Artificial Intelligence is a field of computer science focused on developing systems that perform tasks commonly associated with human intelligence.

Examples:
- Chatbots and virtual assistants.
- Image and speech recognition.
- Recommendation systems.
- Fraud detection.
- Autonomous vehicles.

## 3. What Is Machine Learning (ML)?

Machine Learning is a subset of AI that enables computers to learn patterns from data and make predictions without explicitly programming every rule.

**Example:** An email spam classifier learns from previously labeled emails to identify whether a new email is spam.

Applications:
- Fraud detection.
- House-price prediction.
- Email classification.
- Customer segmentation.
- Product recommendations.

## 4. What Is Deep Learning (DL)?

Deep Learning is a subset of Machine Learning that uses artificial neural networks with multiple layers to learn complex patterns.

Applications:
- Facial recognition.
- Speech recognition.
- Medical image analysis.
- Natural language processing.
- Generative AI.

### Difference Between AI, ML, and DL

| Concept | Meaning | Example |
|---|---|---|
| AI | Broad field of intelligent systems | Virtual assistant |
| ML | Learns patterns from data | Spam classifier |
| Deep Learning | Uses multilayer neural networks | Image recognition |

**Relationship:** Deep Learning is part of Machine Learning, which is part of Artificial Intelligence.

`AI → ML → Deep Learning`

## 5. Types of Machine Learning

There are three major learning approaches:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

### 5.1 Supervised Learning

Supervised Learning uses labeled data. The dataset contains input features and a known target value.

The model learns from these examples and predicts the target for new data.

Example:

| Study Hours | Attendance (%) | Result |
|---:|---:|---|
| 2 | 60 | Fail |
| 5 | 80 | Pass |
| 7 | 95 | Pass |

The model learns from study hours and attendance to predict the result.

**Main tasks:**

- **Classification:** Predicts categories, such as fraud/not fraud or pass/fail.
- **Regression:** Predicts numerical values, such as house prices or sales revenue.

Common algorithms:
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Neural Networks

### 5.2 Unsupervised Learning

Unsupervised Learning uses data without predefined target labels. The model discovers patterns, groups, or relationships in the data.

Example: A business groups customers according to their purchasing behavior without providing predefined customer-group labels.

Common tasks:
- **Clustering:** Groups similar data points.
- **Dimensionality Reduction:** Reduces the number of features while retaining useful information.
- **Association Analysis:** Identifies relationships between items.

Common algorithms:
- K-Means
- DBSCAN
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- Apriori

Applications:
- Customer segmentation.
- Grouping similar documents.
- Discovering purchasing patterns.
- Exploring unusual transactions.

### 5.3 Reinforcement Learning

Reinforcement Learning trains an agent to make decisions by interacting with an environment and receiving rewards or penalties.

The agent learns a strategy that aims to maximize cumulative reward.

Important terms:
- **Agent:** The learner or decision-maker.
- **Environment:** The world in which the agent operates.
- **State:** The current situation.
- **Action:** A decision taken by the agent.
- **Reward:** Feedback about an outcome.
- **Policy:** The strategy used to select actions.

Example: A game-playing agent learns to select better moves by receiving rewards for successful outcomes.

Applications:
- Robotics.
- Game-playing systems.
- Traffic signal optimization.
- Resource allocation.

## 6. Understanding Dataset Terminology

A dataset is a collection of data used for analysis or Machine Learning. It is often organized into rows and columns.

Example:

| Study Hours | Attendance | Marks |
|---:|---:|---:|
| 2 | 60 | 45 |
| 5 | 80 | 75 |
| 7 | 95 | 92 |

Here, Study Hours and Attendance are input features, while Marks is the target to predict.

### 6.1 Dataset

A collection of examples or observations used for analysis or model development.

Example: A CSV file containing 10,000 customer records.

### 6.2 Sample / Instance / Observation

A single example in a dataset, usually represented by one row.

Example: One customer's transaction record.

### 6.3 Feature / Input Variable (X)

A property or measurement used by a model to make predictions.

Examples:
- Age
- Salary
- Transaction amount
- Study hours

Features are commonly represented by `X`.

### 6.4 Label / Target Variable (y)

The value a supervised learning model is trained to predict.

Examples:
- Fraud or not fraud.
- Pass or fail.
- House price.

The target is commonly represented by `y`.

**Important:** The target is provided as the correct answer during training, but it should not also be included as an input feature.

```python
X = df[["Amount", "Transaction_Hour"]]
y = df["Is_Fraud"]
```

Here, `X` contains the input features and `y` contains the target.

### 6.5 Numerical and Categorical Data

**Numerical data:** Values expressed as numbers, such as age, price, or temperature.

**Categorical data:** Values representing categories, such as city, payment method, or product type.

Categorical data may need to be encoded before being used by certain ML algorithms.

### 6.6 Training Dataset

The data used to fit a model and learn patterns.

### 6.7 Validation Dataset

Data used to compare models, tune hyperparameters, and make model-development decisions.

### 6.8 Test Dataset

Data reserved for final evaluation. It should not be repeatedly used to tune the model.

### 6.9 Train, Validation, and Test Split

A common starting point is:
- Training: 70%
- Validation: 15%
- Testing: 15%

Another option is 80% training, 10% validation, and 10% testing.

The appropriate split depends on dataset size, class balance, and the problem. Time-dependent datasets may require chronological splitting.

### 6.10 Data Preprocessing

Preparing raw data before model training.

Common steps:
- Handling missing values.
- Removing duplicates when appropriate.
- Correcting invalid data.
- Encoding categorical variables.
- Scaling numerical features when needed.
- Handling outliers appropriately.

### 6.11 Missing Values

Missing values are data points that are unavailable or were not recorded.

Common approaches:
- Remove rows or columns when justified.
- Fill numerical values using the mean or median.
- Fill categorical values using the mode or an `Unknown` category.

Fit preprocessing steps using training data only, then apply the fitted transformations to validation and test data.

### 6.12 Class Imbalance

Class imbalance occurs when one class contains substantially more examples than another.

Example: A fraud dataset contains 99% legitimate transactions and 1% fraudulent transactions.

Accuracy alone can be misleading in this situation. Precision, recall, F1-score, PR-AUC, and a confusion matrix can provide additional insight.

## 7. Important Machine Learning Terms

### 7.1 Algorithm

A method used to learn patterns from data.

Examples: Decision Tree, K-Means, and Linear Regression.

### 7.2 Model

The learned result of a training algorithm, used to make predictions on new inputs.

### 7.3 Model Training

The process of learning patterns or estimating model parameters from training data.

### 7.4 Prediction / Inference

Using a trained model to generate an output for new input data.

### 7.5 Parameters

Values learned by the model during training.

Example: The weights in a linear regression model or neural network.

### 7.6 Hyperparameters

Settings chosen before or during model development that control model structure or the learning process.

Examples:
- Learning rate.
- Number of trees.
- Maximum tree depth.
- Number of clusters in K-Means.

### 7.7 Epoch

One complete pass through the training dataset during neural network training.

### 7.8 Batch Size

The number of training examples processed together before the model updates its parameters.

### 7.9 Learning Rate

A setting that controls the size of parameter updates during optimization.

### 7.10 Overfitting

Overfitting occurs when a model learns training data too closely, including noise or accidental patterns, and performs poorly on unseen data.

Possible solutions:
- Collect more representative training data.
- Use regularization.
- Reduce model complexity.
- Apply early stopping.
- Use appropriate cross-validation.

### 7.11 Underfitting

Underfitting occurs when a model is too simple or has not learned enough to capture the underlying patterns.

Possible solutions:
- Choose a more suitable model.
- Add informative features.
- Reduce excessive regularization.
- Train longer when appropriate.

### 7.12 Generalization

The ability of a trained model to perform well on previously unseen data.

### 7.13 Data Leakage

Data leakage occurs when information that should not be available during training or model selection influences the learning process or evaluation.

This can make a model appear more accurate than it will be on genuinely new data.

**Example 1: Target leakage**

Suppose a fraud detection model predicts `Is_Fraud`.

```python
# Incorrect: the target is also included in the inputs
X = df[["Amount", "Transaction_Hour", "Is_Fraud"]]
y = df["Is_Fraud"]
```

The model receives the answer it is supposed to predict. This is target leakage.

Correct approach:

```python
X = df[["Amount", "Transaction_Hour"]]
y = df["Is_Fraud"]
```

**Example 2: Preprocessing leakage**

Suppose you calculate a feature's mean or fit a scaler using the entire dataset before splitting it. Information from the validation or test set may influence preprocessing.

Correct approach:
1. Split the data first.
2. Fit the imputer, scaler, and encoder using training data only.
3. Apply the fitted transformations to validation and test data.

**Example 3: Future information leakage**

Suppose you use tomorrow's account status to predict whether today's transaction is fraudulent. That information would not be available at prediction time.

**How to prevent data leakage:**
- Keep the target out of input features.
- Remove features that reveal the target indirectly.
- Split data before fitting preprocessing steps.
- Do not use the test set for model tuning.
- Respect time ordering when predicting future events.

**Remember:** The target variable should be used as the correct answer during supervised training, but not as an input feature when predicting that same target.

### 7.14 Evaluation Metrics

Metrics measure model performance.

**Classification metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

**Regression metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

The appropriate metric depends on the problem and the cost of different types of errors.

## 8. Basic Machine Learning Workflow

A typical ML project follows these steps:

1. Define the problem.
2. Collect or load the dataset.
3. Explore and understand the data.
4. Clean the data and handle missing values.
5. Perform feature engineering and preprocessing.
6. Split data into training, validation, and test sets.
7. Select an appropriate algorithm.
8. Train the model.
9. Tune hyperparameters using training and validation data.
10. Evaluate the final model on the test set.
11. Save and deploy the model.
12. Monitor performance after deployment.

## 9. Recommended Tools and Libraries

- **Python:** Programming language used for many ML projects.
- **NumPy:** Numerical computing and array operations.
- **Pandas:** Data manipulation and analysis.
- **Matplotlib:** Data visualization.
- **Seaborn:** Statistical visualization.
- **Scikit-learn:** Preprocessing, ML algorithms, and evaluation.
- **TensorFlow / Keras:** Deep Learning and neural networks.
- **XGBoost:** Gradient-boosted decision trees.

## 10. Learning Objectives

After studying these foundations, you should be able to:

- Explain AI, ML, and Deep Learning.
- Identify supervised, unsupervised, and reinforcement learning problems.
- Distinguish classification from regression.
- Understand datasets, features, labels, and data splits.
- Explain preprocessing, training, and inference.
- Recognize overfitting, underfitting, and data leakage.
- Select suitable evaluation metrics for basic ML problems.
- Understand the standard ML project workflow.

## Conclusion

AI is the broader field of intelligent systems, Machine Learning enables systems to learn patterns from data, and Deep Learning uses multilayer neural networks to learn complex representations.

Understanding learning approaches, dataset terminology, preprocessing, evaluation, and common pitfalls provides a strong foundation for developing reliable Machine Learning projects.
