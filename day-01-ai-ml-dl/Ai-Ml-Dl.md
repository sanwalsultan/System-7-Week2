# Day 1 — What Is AI, ML and Deep Learning; Learning Types; Dataset Terms

## 1. Introduction

Day 1 covers the meaning of Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), the three major learning approaches, and the vocabulary used to describe datasets.

## 2. What Is Artificial Intelligence (AI)?

Artificial Intelligence is a field of computer science focused on building systems that perform tasks associated with human intelligence, such as reasoning, recognizing patterns, learning, and decision-making.

Examples of AI systems:
- Virtual assistants and chatbots.
- Image and speech recognition.
- Recommendation systems.
- Fraud detection.
- Autonomous vehicles.

## 3. What Is Machine Learning (ML)?

Machine Learning is a subset of AI in which systems learn patterns from data instead of following only explicitly programmed rules.

**Example:** A spam classifier learns from previously labeled emails and then predicts whether a new email is spam.

Common applications:
- Fraud detection.
- House-price prediction.
- Email classification.
- Customer segmentation.
- Product recommendations.

## 4. What Is Deep Learning (DL)?

Deep Learning is a subset of ML that uses artificial neural networks with multiple layers to learn complex patterns and representations from large amounts of data.

Applications:
- Facial recognition.
- Speech recognition.
- Medical image analysis.
- Natural language processing.
- Generative AI.

### Difference Between AI, ML, and Deep Learning

| Concept | Meaning | Example |
|---|---|---|
| AI | Broad field of intelligent systems | Virtual assistant |
| ML | Learns patterns from data | Spam classifier |
| Deep Learning | Uses multilayer neural networks | Image recognition |

**Relationship:** Deep Learning is part of Machine Learning, which is part of Artificial Intelligence.

`AI → ML → Deep Learning`

## 5. Types of Machine Learning

The three major learning approaches are:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

### 5.1 Supervised Learning

Supervised Learning uses labeled data. Each example contains input features and a known target value. The model learns from these examples and predicts the target for new data.

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
- K-Nearest Neighbors (KNN)
- Neural Networks

### 5.2 Unsupervised Learning

Unsupervised Learning uses data without predefined target labels. The model discovers structure, groups, or relationships on its own.

**Example:** A business groups customers by purchasing behavior without providing predefined group labels.

Common tasks:
- **Clustering:** Groups similar data points, for example K-Means and DBSCAN.
- **Dimensionality Reduction:** Reduces the number of features while keeping useful information, for example PCA.
- **Association Analysis:** Finds relationships between items, for example the Apriori algorithm.

Applications:
- Customer segmentation.
- Grouping similar documents.
- Discovering purchasing patterns.
- Detecting unusual transactions.

### 5.3 Reinforcement Learning

Reinforcement Learning trains an agent to make decisions by interacting with an environment and receiving rewards or penalties. The agent learns a policy that maximizes cumulative reward over time.

Important terms:
- **Agent:** The learner or decision-maker.
- **Environment:** The world the agent operates in.
- **State:** The current situation of the agent.
- **Action:** A decision taken by the agent.
- **Reward:** Feedback received after an action.
- **Policy:** The strategy used to choose actions.

**Example:** A game-playing agent learns better moves by receiving rewards for successful outcomes.

Applications:
- Robotics.
- Game-playing systems.
- Traffic signal optimization.
- Resource allocation.

### 5.4 Comparing the Three Approaches

| Approach | Data | Goal | Example |
|---|---|---|---|
| Supervised | Labeled | Predict target values | Spam classifier |
| Unsupervised | Unlabeled | Find structure | Customer segmentation |
| Reinforcement | Rewards from environment | Learn a policy | Game-playing agent |

## 6. Dataset Terms

A dataset is a collection of data used for analysis or model development, usually organized as rows and columns.

Example:

| Study Hours | Attendance | Marks |
|---:|---:|---:|
| 2 | 60 | 45 |
| 5 | 80 | 75 |
| 7 | 95 | 92 |

Here, Study Hours and Attendance are input features, and Marks is the target to predict.

### 6.1 Dataset

A collection of examples or observations used for analysis or model development.

Example: A CSV file containing 10,000 customer records.

### 6.2 Sample / Instance / Observation

A single example in a dataset, usually represented by one row.

Example: One customer's transaction record.

### 6.3 Feature / Input Variable (X)

A property or measurement used by the model to make predictions.

Examples: age, salary, transaction amount, study hours.

Features are commonly represented by `X`.

### 6.4 Label / Target Variable (y)

The value a supervised learning model is trained to predict.

Examples: fraud or not fraud, pass or fail, house price.

The target is commonly represented by `y`.

**Important:** The target is the correct answer during training and must not also be included as an input feature.

```python
X = df[["Amount", "Transaction_Hour"]]
y = df["Is_Fraud"]
```

Here `X` contains the input features and `y` contains the target.

### 6.5 Numerical and Categorical Data

**Numerical data:** Values expressed as numbers, such as age, price, or temperature.

**Categorical data:** Values representing categories, such as city, payment method, or product type.

Categorical data often needs encoding before it can be used by ML algorithms.

### 6.6 Train, Validation, and Test Sets

- **Training set:** Data used to fit the model and learn patterns.
- **Validation set:** Data used to compare models and tune hyperparameters during development.
- **Test set:** Data reserved for final evaluation only.

A common starting split:
- Training: 70%
- Validation: 15%
- Testing: 15%

Or 80% training, 10% validation, 10% testing.

### 6.7 Other Important Terms

- **Algorithm:** A method used to learn patterns from data, such as Linear Regression or K-Means.
- **Model:** The learned result of training, used to make predictions on new inputs.
- **Training:** The process of estimating model parameters from data.
- **Prediction / Inference:** Using a trained model to produce outputs for new inputs.
- **Parameters:** Values learned during training, such as weights.
- **Hyperparameters:** Settings chosen before training, such as learning rate or number of trees.
- **Overfitting:** The model learns training data (including noise) too closely and performs poorly on new data.
- **Underfitting:** The model is too simple to capture the underlying patterns.
- **Generalization:** The ability of a model to perform well on previously unseen data.

## 7. Learning Objectives

After completing Day 1, you should be able to:

- Explain the difference between AI, Machine Learning, and Deep Learning.
- Identify supervised, unsupervised, and reinforcement learning problems.
- Distinguish classification from regression.
- Define dataset, sample, feature, label, and train/validation/test split.
- Keep the target variable out of the input features.

## Conclusion

AI is the broader field of intelligent systems, ML enables systems to learn from data, and DL uses multilayer neural networks to learn complex representations. Choosing the right learning approach and understanding dataset terminology are the first steps in any ML project.
