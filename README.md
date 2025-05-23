
---

# Decision Tree Classifier: Social Network Ads Prediction 💭  ![image](https://github.com/user-attachments/assets/2baad095-c755-4c36-acc8-e209d2f495c9) ![image](https://github.com/user-attachments/assets/435d578c-500f-4693-ba29-4a5cf489b5ab) ![image](https://github.com/user-attachments/assets/468c9f7f-d6f6-446f-925c-49eb0b8b46c7)


# Decision Tree: A Comprehensive Guide

## Introduction

A **Decision Tree** is a popular supervised machine learning algorithm that is used for both classification and regression tasks. It works by breaking down complex decision-making processes into a series of simpler decisions, represented as a tree-like graph of nodes and branches. Decision Trees are intuitive, easy to visualize, and require minimal data preparation.

---

## What Is a Decision Tree?

A **Decision Tree** is a flowchart-like structure where:

- **Internal nodes** represent "tests" or "decisions" on attributes/features.
- **Branches** represent the outcome of a test.
- **Leaf nodes** represent the class label (for classification) or value (for regression).

The path from the root to a leaf represents a classification or decision rule.

---

## How Decision Trees Work

1. **Select the Best Feature**: The algorithm chooses the feature that best splits the dataset into subsets with distinct target values. Common criteria:
   - **Gini Impurity** (for classification)
   - **Entropy/Information Gain** (for classification)
   - **Mean Squared Error** (for regression)

2. **Split the Dataset**: Divide the dataset into subsets based on the selected feature.

3. **Repeat Recursively**: For each subset, repeat the process until one of the stopping criteria is met (e.g., all samples in a subset belong to the same class, or maximum depth is reached).

4. **Assign Output**: Assign a class (for classification) or value (for regression) to each leaf node.

---

## Example: Decision Tree for Classification

Suppose we want to build a Decision Tree to classify whether someone will play tennis based on the weather.

| Outlook | Temperature | Humidity | Windy | Play Tennis |
|---------|-------------|----------|-------|-------------|
| Sunny   | Hot         | High     | False | No          |
| Sunny   | Hot         | High     | True  | No          |
| Overcast| Hot         | High     | False | Yes         |
| Rain    | Mild        | High     | False | Yes         |
| Rain    | Cool        | Normal   | False | Yes         |
| Rain    | Cool        | Normal   | True  | No          |
| Overcast| Cool        | Normal   | True  | Yes         |

**Step 1:** Calculate Information Gain for each feature and choose the best one (e.g., Outlook).

**Step 2:** Split the dataset based on Outlook:
- **Sunny** → Further split based on Humidity.
- **Overcast** → Always Play Tennis = Yes (pure leaf).
- **Rain** → Further split based on Windy.

The resulting tree might look like:

```
Outlook?
├── Sunny
│   └── Humidity?
│       ├── High: No
│       └── Normal: Yes
├── Overcast: Yes
└── Rain
    └── Windy?
        ├── False: Yes
        └── True: No
```

---

## Example: Decision Tree for Regression

Suppose you want to predict house prices based on features like size and location.

- At each split, the algorithm chooses the feature and threshold that minimizes the variance (mean squared error) in the target variable (house price).
- Leaf nodes contain the average house price of the subset.

---

## Advantages of Decision Trees

- **Easy to understand and interpret**: Can be visualized graphically.
- **No need for feature scaling**: Handles both numerical and categorical data.
- **Handles non-linear relationships**: No need for linearity in the data.

---

## Disadvantages of Decision Trees

- **Prone to overfitting**: Especially with deep trees and small datasets.
- **Unstable**: Small variations in the data can result in a different tree.
- **Biased towards features with more levels**: Can prefer features with more categories.

---

## Best Practices

- **Prune the tree**: Limit the maximum depth or minimum samples per leaf.
- **Use ensembles**: Techniques like Random Forest and Gradient Boosting combine multiple trees for better generalization.
- **Cross-validation**: Use to select optimal tree parameters.

---

## Implementing a Decision Tree in Python (Scikit-learn Example)

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load sample data
X, y = load_iris(return_X_y=True)

# Initialize and fit classifier
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Visualize the tree
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
plt.show()
```

---


---

# Decision Tree Classifier: Social Network Ads Prediction 💭  ![image](https://github.com/user-attachments/assets/2baad095-c755-4c36-acc8-e209d2f495c9) ![image](https://github.com/user-attachments/assets/435d578c-500f-4693-ba29-4a5cf489b5ab) ![image](https://github.com/user-attachments/assets/468c9f7f-d6f6-446f-925c-49eb0b8b46c7)






This notebook, [`Decision_Tree.ipynb`](https://github.com/vyasdeepti/Machine-Learning/blob/main/Decision_Tree.ipynb), demonstrates a complete machine learning workflow using a **Decision Tree Classifier** to predict user purchase decisions based on social network advertisements. The project illustrates each stage of the pipeline: data import, preprocessing, exploratory data analysis, model training, evaluation, and practical interpretation of results.

---

## 💡 Table of Contents 📚

- 🧪 [Overview](#overview)
- ✨ [What Is a Decision Tree: Working and Example?](#Whatisdecisiontree?)
- 📊 [Dataset](#dataset)
- 🏗️ [Workflow](#workflow)
  - [1. Import Libraries](#1-import-libraries)
  - [2. Import Dataset](#2-import-dataset)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
  - [5. Feature Engineering & Splitting](#5-feature-engineering--splitting)
  - [6. Model Training](#6-model-training)
  - [7. Model Evaluation](#7-model-evaluation)
  - [8. Visualization](#8-visualization)
- 🔎 [How to Run the Notebook](#how-to-run-the-notebook)
- 🧪 [Results & Interpretation](#results--interpretation)
- 🛠️ [Requirements](#requirements)
- ✨ [References](#references)

---

## Overview

The notebook guides you through a supervised classification problem: **Will a social network user purchase a product after seeing an ad?** Using the `DecisionTreeClassifier` from scikit-learn, we build a predictive model based on user demographics and salary information. This workflow is suitable for students, data science beginners, and anyone seeking a practical illustration of decision trees in Python.

---

## Dataset 📄

- **File:** `Social_Network_Ads.csv`
- **Columns:**
  - `User ID` (removed in preprocessing)
  - `Gender` (categorical)
  - `Age` (numerical)
  - `EstimatedSalary` (numerical)
  - `Purchased` (target: 0 = No, 1 = Yes)

The dataset consists of 400 entries, each representing a unique user and their response to an online advertisement.

---

## Workflow 📗

### 1. Import Libraries

The notebook starts by importing all the necessary libraries, including:
- Data manipulation: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Preprocessing and modeling: `scikit-learn`

### 2. Import Dataset

- Loads `Social_Network_Ads.csv` into a pandas DataFrame.
- Displays the first few rows to understand the structure.

### 3. Data Preprocessing

- **Drop Irrelevant Columns:** Removes `User ID` as it does not contribute to prediction.
- **Categorical Encoding:** Transforms `Gender` into a numeric format using label encoding.
- **Feature Scaling (Optional):** Scales `Age` and `EstimatedSalary` for improved model performance.
- **Null/Outlier Check:** (Recommended for real-world data)

### 4. Exploratory Data Analysis (EDA) 🚀

- Uses `describe()` to summarize numerical features (mean, std, min/max, quartiles).
- Visualizes distributions (e.g., histograms, boxplots) and examines relationships between features.

 
### 5. Feature Engineering & Splitting

- **Feature Selection:** Chooses relevant columns as features (`Gender`, `Age`, `EstimatedSalary`).
- **Train-Test Split:** Splits the data into training and testing sets (commonly 75% train, 25% test) using `train_test_split`.

### 6. Model Training

- Initializes the **Decision Tree Classifier** (`DecisionTreeClassifier` from scikit-learn).
- Fits the model to the training data.

### 7. Model Evaluation

- Predicts outcomes on the test set.
- Calculates metrics:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report** (precision, recall, f1-score)
  - **F1 Score**
- Optionally, plots Precision-Recall curves and other metrics.

### 8. Visualization 📊

- Visualizes the decision boundaries of the trained classifier.
- Uses scatter plots to show correctly and incorrectly classified points.
- Optionally, plots the tree structure for interpretability.

 
 


---

## 🏗️ Project Structure

```plaintext
notebooks/
  └── Decision_Tree.ipynb
data/
  └── Social_Network_Ads.csv
README.md

```

## How to Run the Notebook

1. **Clone the Repository** or download the notebook file.
    ```bash
    git clone https://github.com/vyasdeepti/Machine-Learning.git
    cd Machine-Learning
    ```
2. **Install Required Libraries** (if not already installed):
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3. **Place the CSV File**  
   Ensure `Social_Network_Ads.csv` is in the same directory as the notebook.

4. **Start Jupyter Notebook or Colab**
    - For Jupyter:  
      ```bash
      jupyter notebook
      ```
      Then open `Decision_Tree.ipynb`.
    - For Google Colab: Use the Colab badge at the top of the notebook or upload the notebook directly.

5. **Run All Cells**  
   Follow the notebook top-down, executing each cell in order.

---

## Results & Interpretation

Here's an explanation of the concepts in the context of the code in Decision_Tree.ipynb:

---

### 1. Label Encoding
**What it is:** Label encoding is the process of converting categorical variables (like "Gender") into numeric codes, so they can be used in machine learning models.

**How it's done in the code:**  
This code uses `LabelEncoder` from scikit-learn, likely doing something like:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_net['Gender'] = le.fit_transform(df_net['Gender'])
```
This converts the "Gender" column from "Male"/"Female" to 1/0 (or 0/1), making it usable for the decision tree model.

---

### 2. Correlation Matrix
**What it is:** A correlation matrix shows the relationship (correlation coefficient) between pairs of features. Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).

**How it's used:**  
This code probably uses pandas or seaborn to visualize the correlations:
```python
corr = df_net.corr()
sns.heatmap(corr, annot=True)
```
This helps you see which features are strongly related to each other or to the target "Purchased".
![image](https://github.com/user-attachments/assets/5681b839-e619-4a97-aaa5-bd398d0a246f)


---

### 3. Feature Scaling
**What it is:** Feature scaling standardizes numeric features so that they have similar ranges, which helps many machine learning models perform better.

**How it's done in the code:**  
We have used `StandardScaler`:
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
```
This transforms features like "Age" and "EstimatedSalary" to have mean 0 and standard deviation 1.

---

### 4. Confusion Matrix
**What it is:** A confusion matrix is a table that visualizes the performance of a classification algorithm, showing counts of true positives, false positives, true negatives, and false negatives.

 ![image](https://github.com/user-attachments/assets/f7155058-488b-4cea-831b-7dd719f9aa94)

**How it's used in the code:**  
We have used scikit-learn’s `confusion_matrix`:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```
The matrix lets us see how many correct and incorrect predictions the model made for each class (Purchased = 1 or 0).

---

**Summary Table:**

| Concept           | Purpose                                           | Code Example                                                 |
|-------------------|---------------------------------------------------|--------------------------------------------------------------|
| Label Encoding    | Convert categories to numbers                     | `df_net['Gender'] = le.fit_transform(df_net['Gender'])`      |
| Correlation Matrix| Show relationships between features               | `corr = df_net.corr(); sns.heatmap(corr, annot=True)`        |
| Feature Scaling   | Standardize feature ranges                        | `X = sc.fit_transform(X)`                                    |
| Confusion Matrix  | Evaluate classifier predictions                   | `cm = confusion_matrix(y_test, y_pred)`                      |


Upon completion, you will have: 🚀

- A well-trained Decision Tree model for the classification problem.
- Performance metrics showing how well the model predicts user purchases.
- Visualizations that make the model’s logic and performance transparent.
- Insights into which features are most important for the prediction.

---

## Requirements 🔎

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## References ✨

- [scikit-learn Documentation: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---
## ❓ FAQ

**Q:** What Python version is required?  
**A:** Python 3.7 or higher.

**Q:** Can I use my own dataset?  
**A:** Yes! Replace `Social_Network_Ads.csv` with your data.

---

*For questions or suggestions, please open an issue or contact the repository maintainer via GitHub.*

Let me know if you want this tailored for a classroom audience, a different dataset, or expanded with troubleshooting tips!
