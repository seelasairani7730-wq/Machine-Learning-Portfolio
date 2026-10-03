# Customer Churn Prediction using Logistic Regression

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Predicting churn can help businesses identify customers who may be at risk of leaving and support data-driven retention strategies.

This project builds a **Customer Churn Prediction system using Logistic Regression**. The model predicts whether a customer is likely to **churn or stay** based on customer information such as tenure, contract type, monthly charges, payment method, internet service, and other service-related attributes.

The project demonstrates a complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, interpretation, and model persistence.

---

## Problem Statement

The objective of this project is to build a binary classification model that predicts whether a customer will:

* **0 → Stay**
* **1 → Churn**

Accurately identifying customers who may churn can help businesses investigate potential retention opportunities.

---

## Machine Learning Algorithm

**Logistic Regression**

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification problems.

In this project, the model estimates the probability of customer churn and converts that probability into a binary prediction.

---

## Dataset

The project uses the **Telco Customer Churn dataset**.

* **Total records after preprocessing:** 7,032
* **Original records:** 7,043
* **Target variable:** `Churn`
* **Problem type:** Binary Classification

The dataset contains information about customer demographics, account details, subscribed services, contract type, payment method, and billing information.

### Target Distribution

| Class | Customers | Percentage |
| ----- | --------: | ---------: |
| Stay  |     5,163 |     73.42% |
| Churn |     1,869 |     26.58% |

---

## Features

### Numerical Features

* SeniorCitizen
* tenure
* MonthlyCharges
* TotalCharges

### Categorical Features

* gender
* Partner
* Dependents
* PhoneService
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* Contract
* PaperlessBilling
* PaymentMethod

The `customerID` column was removed because it is an identifier and does not provide useful predictive information.

---

## Exploratory Data Analysis

Several relationships between customer characteristics and churn were explored.

### Contract Type

Customers with month-to-month contracts showed a higher observed churn rate than customers with one-year or two-year contracts.

### Tenure

Customers who churned had a lower average and median tenure compared with customers who stayed.

* Average tenure of customers who stayed: **37.65 months**
* Average tenure of customers who churned: **17.98 months**

### Monthly Charges

Customers who churned had higher average monthly charges than customers who stayed.

* Average monthly charges for customers who stayed: **61.31**
* Average monthly charges for customers who churned: **74.44**

### Payment Method

Electronic check customers showed the highest observed churn rate among the payment methods in the dataset.

These findings represent associations observed in the dataset and should not be interpreted as evidence that these factors directly cause churn.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed the `customerID` identifier.
2. Converted `TotalCharges` from object to numeric format.
3. Identified 11 missing `TotalCharges` values caused by blank entries.
4. Removed those 11 incomplete records.
5. Separated numerical and categorical features.
6. Standardized numerical features using `StandardScaler`.
7. Encoded categorical features using `OneHotEncoder`.
8. Used `ColumnTransformer` to apply the appropriate preprocessing to each feature type.
9. Used an 80/20 train-test split with stratification.

The preprocessing steps were fitted only on the training data to avoid data leakage.

---

## Model Training

A Logistic Regression classifier was trained using the preprocessed training data.

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The final preprocessing and model were combined into a single **Scikit-learn Pipeline**.

This allows new raw customer data to pass through the same preprocessing steps automatically before prediction.

---

## Model Evaluation

The model was evaluated on the unseen test dataset containing **1,407 customers**.

### Results

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **80.53%** |
| Churn Precision |    **65%** |
| Churn Recall    |    **57%** |
| Churn F1-score  |    **61%** |
| ROC-AUC         | **0.8361** |

The model achieved an accuracy of approximately **80.53%**.

For the Churn class, the model achieved a precision of **0.65**, recall of **0.57**, and F1-score of **0.61**.

The ROC-AUC score of **0.8361** indicates that the model has useful ability to distinguish between customers who churn and customers who stay across different classification thresholds.

---

## Confusion Matrix

The confusion matrix was:

```text
                Predicted
              Stay   Churn

Actual Stay    918    115
Actual Churn   159    215
```

This means:

* **918** customers who stayed were correctly classified.
* **215** customers who churned were correctly classified.
* **115** customers who stayed were incorrectly classified as churn.
* **159** customers who churned were incorrectly classified as stay.

---

## Model Interpretability

Logistic Regression provides coefficients that can be used to understand how features are associated with the model's predicted churn probability.

Examples of features with relatively large coefficient magnitudes included:

| Feature                        | Coefficient |
| ------------------------------ | ----------: |
| Contract — Two year            |     -1.3648 |
| Tenure                         |     -1.3574 |
| Internet Service — Fiber optic |      1.1074 |
| Contract — One year            |     -0.7501 |
| Total Charges                  |      0.6455 |

A positive coefficient is associated with a higher predicted probability of churn, while a negative coefficient is associated with a lower predicted probability of churn, holding the other model features constant.

These coefficients describe model associations and should not be interpreted as causal relationships.

---

## Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Train-Test Split
     ↓
Numerical Scaling
     +
Categorical Encoding
     ↓
ColumnTransformer
     ↓
Logistic Regression
     ↓
Predictions
     ↓
Model Evaluation
     ↓
Save Complete Pipeline
```

---

## Project Structure

```text
Customer_Churn_Prediction/
│
├── README.md
├── Customer_Churn_Prediction.ipynb
├── customer_churn_logistic_regression.pkl
├── requirements.txt
│
└── images/
    ├── churn_distribution.png
    ├── contract_churn.png
    ├── tenure_churn.png
    ├── payment_method_churn.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── logistic_coefficients.png
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Google Colab
* Jupyter Notebook

---

## How to Run

### 1. Clone the repository

```bash
git clone <seelasairani7730-wq>
cd Customer_Churn_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
Customer_Churn_Prediction.ipynb
```

and run the notebook cells.

### 4. Use the trained model

The trained pipeline is available as:

```text
customer_churn_logistic_regression.pkl
```

The saved pipeline includes both preprocessing and the Logistic Regression model.

---

## Key Learnings

Through this project, I practiced:

* Binary classification using Logistic Regression
* Exploratory data analysis
* Handling missing values
* One-hot encoding
* Feature scaling
* Using `ColumnTransformer`
* Preventing data leakage
* Train-test splitting with stratification
* Classification metrics
* Confusion matrix analysis
* ROC-AUC evaluation
* Model coefficient interpretation
* Building and saving an end-to-end Scikit-learn pipeline

---

## Future Improvements

Possible improvements include:

* Testing different classification thresholds
* Comparing Logistic Regression with other classification algorithms
* Hyperparameter tuning
* Cross-validation
* Evaluating techniques for handling class imbalance
* Building a web interface for interactive churn prediction
