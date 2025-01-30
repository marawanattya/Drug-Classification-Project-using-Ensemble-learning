# Drug Classification Project using Ensemble learning

## Overview
This project focuses on classifying drugs based on patient characteristics such as age, sex, blood pressure, cholesterol levels, and sodium-to-potassium ratio. The dataset used is the **Drug Classification Dataset** from Kaggle, which contains 200 instances and 6 features. The goal is to predict the appropriate drug type for a patient using various machine learning models and ensemble techniques.

---

## Dataset
The dataset used in this project is **drug200.csv**, which contains the following features:
- **Age**: Age of the patient
- **Sex**: Gender of the patient (Male/Female)
- **BP**: Blood pressure level (High/Normal/Low)
- **Cholesterol**: Cholesterol level (High/Normal)
- **Na_to_K**: Sodium-to-potassium ratio in blood
- **Drug**: Drug type (target variable)

---

## Preprocessing
1. **Handling Missing Values**: The dataset has no missing values.
2. **Encoding Categorical Variables**: Categorical features like `Sex`, `BP`, `Cholesterol`, and `Drug` were encoded using `LabelEncoder`.
3. **Feature Scaling**: Numerical features (`Age` and `Na_to_K`) were standardized using `StandardScaler`.
4. **Train-Test Split**: The dataset was split into training (80%) and testing (20%) sets.

---

## Exploratory Data Analysis (EDA)
Several visualizations were created to understand the dataset:
1. **Age Distribution**: A histogram showing the distribution of patient ages.
2. **Na_to_K Distribution**: A histogram showing the distribution of sodium-to-potassium ratios.
3. **Drug Count**: A count plot showing the frequency of each drug type.
4. **Gender Distribution**: A count plot showing the distribution of male and female patients.
5. **BP and Cholesterol Distribution**: Count plots showing the distribution of blood pressure and cholesterol levels.
6. **Pairplot**: A pairplot to visualize relationships between features, colored by drug type.
7. **Correlation Heatmap**: A heatmap showing correlations between numerical features.
8. **Boxplots**: Boxplots showing the distribution of `Age` and `Na_to_K` across different drug types.

---

## Models and Ensemble Techniques
The following machine learning models and ensemble techniques were implemented:
1. **Base Models**:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors (KNN)

2. **Ensemble Techniques**:
   - **Max Voting**: Combines predictions from multiple models by majority voting.
   - **Averaging**: Averages the predicted probabilities from multiple models.
   - **Weighted Averaging**: Averages the predicted probabilities with assigned weights.
   - **Bagging**: Uses bootstrap aggregation with Decision Trees.
   - **Random Forest**: An ensemble of Decision Trees with bagging and feature randomness.
   - **AdaBoost**: Adaptive Boosting, which focuses on correcting errors from previous models.
   - **Gradient Boosting**: Builds models sequentially to correct residual errors.
   - **XGBoost**: An optimized implementation of Gradient Boosting with regularization.
   - **Stacking**: Combines predictions from multiple models using a meta-classifier (Logistic Regression).

---

## Evaluation Metrics
The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

The results are summarized below:

| Method               | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Max Voting           | 1.000    | 1.000     | 1.000  | 1.000    |
| Averaging            | 1.000    | 1.000     | 1.000  | 1.000    |
| Weighted Averaging   | 1.000    | 1.000     | 1.000  | 1.000    |
| Bagging              | 1.000    | 1.000     | 1.000  | 1.000    |
| Random Forest        | 1.000    | 1.000     | 1.000  | 1.000    |
| Gradient Boosting    | 1.000    | 1.000     | 1.000  | 1.000    |
| Stacking             | 1.000    | 1.000     | 1.000  | 1.000    |
| XGBoost              | 0.975    | 0.977     | 0.975  | 0.974    |
| AdaBoost             | 0.800    | 0.664     | 0.800  | 0.719    |

---

## Results
- **Best Technique**: **Max Voting**, **Averaging**, **Weighted Averaging**, **Bagging**, **Random Forest**, **Gradient Boosting**, and **Stacking** achieved **100% accuracy** on the test set.
- **XGBoost** performed slightly worse with an accuracy of **97.5%**.
- **AdaBoost** had the lowest accuracy of **80%**.

---

## Visualization of Results
A bar plot comparing the accuracy, precision, recall, and F1 score of all ensemble methods is provided in the notebook.

---

## How to Run the Code
1. Install the required libraries:
   ```bash
   pip install kaggle pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
2. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d prathamtripathi/drug-classification
   ```
3. Unzip the dataset:
   ```bash
   unzip drug-classification.zip
   ```
4. Run the Jupyter Notebook or Python script to execute the code.

---

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `kaggle`
