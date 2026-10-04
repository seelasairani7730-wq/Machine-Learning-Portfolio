# Wine Classification using K-Nearest Neighbors (KNN)

## Project Overview

This project implements a **K-Nearest Neighbors (KNN)** classification model to classify wine samples into one of three wine classes based on their chemical characteristics.

The project demonstrates a complete machine learning workflow, including exploratory data analysis, feature scaling, hyperparameter selection using cross-validation, model evaluation, and model persistence.

---

## Problem Statement

The objective is to build a machine learning model that can classify a wine sample into one of three classes based on its chemical properties.

KNN is a **distance-based algorithm**, so feature scaling is particularly important for this problem.

---

## Machine Learning Algorithm

**K-Nearest Neighbors (KNN)**

KNN classifies a new data point by finding the **K closest training samples** and assigning the class that is most common among those neighbors.

The model does not learn explicit mathematical parameters like linear regression. Instead, it relies on the distances between data points during prediction.

---

## Dataset

The project uses the **Wine dataset provided by Scikit-learn**.

| Property       | Value |
| -------------- | ----: |
| Total samples  |   178 |
| Features       |    13 |
| Target classes |     3 |
| Missing values |     0 |

### Target Classes

* Class 0
* Class 1
* Class 2

### Features

The dataset contains chemical measurements such as:

* Alcohol
* Malic Acid
* Ash
* Alcalinity of Ash
* Magnesium
* Total Phenols
* Flavanoids
* Nonflavanoid Phenols
* Proanthocyanins
* Color Intensity
* Hue
* OD280/OD315 of Diluted Wines
* Proline

---

## Project Workflow

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Check Missing Values
     ↓
Analyze Target Classes
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
K Value Selection using Cross-Validation
     ↓
Train KNN Model
     ↓
Make Predictions
     ↓
Evaluate Model
     ↓
Save Model Pipeline
```

---

## Exploratory Data Analysis

### Wine Class Distribution

The dataset contains three wine classes with the following distribution:

| Class   | Samples |
| ------- | ------: |
| Class 0 |      59 |
| Class 1 |      71 |
| Class 2 |      48 |

![Wine Class Distribution](images/wine_class_distribution.png)

The classes are reasonably distributed, with no extremely dominant class.

---

## Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Dataset Split

* Training samples: **142**
* Testing samples: **36**

`stratify=y` was used to maintain a similar class distribution in both training and testing datasets.

---

## Feature Scaling

KNN is a **distance-based algorithm**.

The features in the Wine dataset have very different numerical ranges. For example, `proline` has values much larger than features such as `alcohol`.

Without scaling, features with larger numerical values could have a greater influence on distance calculations.

Therefore, **StandardScaler** was used.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data and then applied to the test data to avoid data leakage.

---

## Selecting the Best K

The value of **K** determines how many neighboring samples are considered when making a prediction.

Instead of selecting K based only on the test set, **5-fold Stratified Cross-Validation** with `GridSearchCV` was used.

Values of K from **1 to 20** were evaluated.

```python
param_grid = {
    "knn__n_neighbors": range(1, 21)
}
```

### Cross-Validation Result

* **Best K: 13**
* **Mean Cross-Validation Accuracy: 96.50%**

![KNN Cross Validation Accuracy](images/knn_cv_accuracy.png)

Using cross-validation provides a more reliable basis for selecting the hyperparameter than choosing K directly from test-set performance.

---

## Model Training

A Scikit-learn Pipeline was used to combine feature scaling and KNN classification.

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])
```

The final model uses:

```text
Algorithm: K-Nearest Neighbors
Best K: 13
```

---

## Model Evaluation

The final model was evaluated on the previously unseen test set.

### Test Accuracy

**100%**

```text
Test Accuracy: 1.00
```

### Classification Report

| Class   | Precision | Recall | F1-Score |
| ------- | --------: | -----: | -------: |
| Class 0 |      1.00 |   1.00 |     1.00 |
| Class 1 |      1.00 |   1.00 |     1.00 |
| Class 2 |      1.00 |   1.00 |     1.00 |

**Overall Accuracy: 100%**

---

## Confusion Matrix

The model correctly classified all **36 test samples**.

```text
[[12  0  0]
 [ 0 14  0]
 [ 0  0 10]]
```

![Confusion Matrix](images/confusion_matrix.png)

This means:

* Class 0 → 12/12 correctly classified
* Class 1 → 14/14 correctly classified
* Class 2 → 10/10 correctly classified
* Total misclassifications → **0**

---

## Model Saving

The complete preprocessing and KNN model were saved together as a Scikit-learn pipeline.

```python
joblib.dump(
    best_knn_model,
    "knn_wine_classification.pkl"
)
```

The saved pipeline includes:

```text
StandardScaler
      +
KNeighborsClassifier
```

This allows new data to be processed using the same scaling procedure before prediction.

The saved model was also reloaded and tested to verify that its predictions matched the original model.

```text
Predictions match: True
```

---

## Project Structure

```text
Wine_Classification/
│
├── KNN_Wine_Classification.ipynb
├── README.md
├── requirements.txt
├── knn_wine_classification.pkl
│
└── images/
    ├── wine_class_distribution.png
    ├── knn_cv_accuracy.png
    └── confusion_matrix.png
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

---

## Key Learnings

* Understanding how **KNN classification** works
* Understanding why **feature scaling** is important for distance-based algorithms
* Using `StandardScaler` correctly without data leakage
* Selecting hyperparameters using **cross-validation**
* Using `GridSearchCV` for systematic model selection
* Evaluating multiclass classification using accuracy, precision, recall, F1-score, and confusion matrix
* Building a reusable Scikit-learn pipeline
* Saving and reloading a trained machine learning model

---

## Limitations

Although the final test accuracy is **100%**, the test set contains only 36 samples and the dataset itself is relatively small.

Therefore, the result should not be interpreted as proof that the model will achieve 100% accuracy on every real-world wine sample.

Testing the model on additional datasets would provide a stronger estimate of its generalization ability.

---

## Future Improvements

* Compare KNN with other classification algorithms such as Decision Tree and Random Forest
* Perform additional hyperparameter tuning
* Evaluate the model using repeated cross-validation
* Test the model on external wine datasets
* Build a simple web application for interactive wine classification

---

## Conclusion

This project demonstrates how **K-Nearest Neighbors** can be used for multiclass wine classification.

The final KNN pipeline selected **K = 13** through 5-fold cross-validation and achieved **96.50% mean cross-validation accuracy** and **100% accuracy on the held-out test set**.

The project also highlights the importance of feature scaling and proper validation when working with distance-based machine learning algorithms.
