
---

# Decision Tree Classifier: Social Network Ads Prediction

This notebook, [`Decision_Tree.ipynb`](https://github.com/vyasdeepti/Machine-Learning/blob/main/Decision_Tree.ipynb), demonstrates a complete machine learning workflow using a **Decision Tree Classifier** to predict user purchase decisions based on social network advertisements. The project illustrates each stage of the pipeline: data import, preprocessing, exploratory data analysis, model training, evaluation, and practical interpretation of results.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
  - [1. Import Libraries](#1-import-libraries)
  - [2. Import Dataset](#2-import-dataset)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
  - [5. Feature Engineering & Splitting](#5-feature-engineering--splitting)
  - [6. Model Training](#6-model-training)
  - [7. Model Evaluation](#7-model-evaluation)
  - [8. Visualization](#8-visualization)
- [How to Run the Notebook](#how-to-run-the-notebook)
- [Results & Interpretation](#results--interpretation)
- [Requirements](#requirements)
- [References](#references)

---

## Overview

The notebook guides you through a supervised classification problem: **Will a social network user purchase a product after seeing an ad?** Using the `DecisionTreeClassifier` from scikit-learn, we build a predictive model based on user demographics and salary information. This workflow is suitable for students, data science beginners, and anyone seeking a practical illustration of decision trees in Python.

---

## Dataset

- **File:** `Social_Network_Ads.csv`
- **Columns:**
  - `User ID` (removed in preprocessing)
  - `Gender` (categorical)
  - `Age` (numerical)
  - `EstimatedSalary` (numerical)
  - `Purchased` (target: 0 = No, 1 = Yes)

The dataset consists of 400 entries, each representing a unique user and their response to an online advertisement.

---

## Workflow

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

### 4. Exploratory Data Analysis (EDA)

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

### 8. Visualization

- Visualizes the decision boundaries of the trained classifier.
- Uses scatter plots to show correctly and incorrectly classified points.
- Optionally, plots the tree structure for interpretability.

---

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

Upon completion, you will have:

- A well-trained Decision Tree model for the classification problem.
- Performance metrics showing how well the model predicts user purchases.
- Visualizations that make the model’s logic and performance transparent.
- Insights into which features are most important for the prediction.

---

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install all requirements using:
```bash
pip install -r requirements.txt
```
*(Create `requirements.txt` or install individually as needed.)*

---

## References

- [scikit-learn Documentation: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---


*For questions or suggestions, please open an issue or contact the repository maintainer via GitHub.*

---

Let me know if you want this tailored for a classroom audience, a different dataset, or expanded with troubleshooting tips!
