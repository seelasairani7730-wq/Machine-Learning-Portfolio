# Car Evaluation Classification using Decision Tree

## Project Overview

This project uses a **Decision Tree Classifier** to predict the overall evaluation of a car based on its buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety level.

The project demonstrates how Decision Trees can learn interpretable decision rules from categorical data and how preprocessing, hyperparameter tuning, and model evaluation can be combined into a complete machine learning workflow.

---

## Problem Statement

The objective is to classify cars into one of four evaluation categories based on their characteristics:

- **Unacceptable**
- **Acceptable**
- **Good**
- **Very Good**

This can help demonstrate how machine learning can be used to automatically evaluate products based on multiple categorical attributes.

---

## Machine Learning Algorithm

### Decision Tree Classifier

A Decision Tree makes predictions by repeatedly splitting data based on feature values.

The resulting tree consists of decision rules that can be interpreted and visualized, making Decision Trees useful when model interpretability is important.

---

## Dataset

The project uses the **Car Evaluation dataset**.

The dataset contains **1,728 car records** and **6 input features**.

### Features

| Feature | Description |
|---|---|
| `buying` | Buying price of the car |
| `maint` | Maintenance cost |
| `doors` | Number of doors |
| `persons` | Passenger capacity |
| `lug_boot` | Luggage boot size |
| `safety` | Safety level |

### Target

`class`

Possible target classes:

- `unacc`
- `acc`
- `good`
- `vgood`

---

## Dataset Distribution

The target classes are imbalanced:

| Class | Number of Samples |
|---|---:|
| Unacceptable (`unacc`) | 1210 |
| Acceptable (`acc`) | 384 |
| Good (`good`) | 69 |
| Very Good (`vgood`) | 65 |

Because of this imbalance, evaluation includes **precision, recall, F1-score, and confusion matrix**, rather than relying only on accuracy.

![Car Evaluation Class Distribution](images/car_class_distribution.png)

---

## Project Workflow

1. Import required libraries
2. Load the Car Evaluation dataset
3. Explore the dataset
4. Check missing values and duplicates
5. Analyze feature distributions
6. Analyze target class distribution
7. Separate features and target
8. Encode categorical features using One-Hot Encoding
9. Split the dataset using stratified sampling
10. Train a baseline Decision Tree
11. Tune Decision Tree hyperparameters using GridSearchCV
12. Evaluate the final model
13. Visualize the Decision Tree
14. Analyze feature importance
15. Save and verify the trained pipeline

---

## Data Preprocessing

All six input features are categorical.

Instead of manually assigning numerical values to categories, **One-Hot Encoding** was used to avoid introducing artificial ordinal relationships between categories.

For example:

```text
low < medium < high
```

should not automatically imply a numerical relationship for every categorical feature.

The preprocessing and model were combined into a single **Scikit-learn Pipeline**.

---

## Train-Test Split

The dataset was divided using:

- **80% Training Data**
- **20% Testing Data**
- `random_state = 42`
- Stratified splitting

### Dataset Sizes

```text
Training samples: 1382
Testing samples: 346
```

Stratification was used to preserve the class distribution in both training and testing datasets.

---

## Baseline Model

A Decision Tree Classifier was first trained without hyperparameter tuning.

### Baseline Accuracy

**97.40%**

The baseline model already achieved strong performance on the test set.

---

## Hyperparameter Tuning

To improve model selection and avoid relying on a single arbitrary tree configuration, **GridSearchCV with 5-fold Stratified Cross-Validation** was used.

The following hyperparameters were evaluated:

```text
max_depth
min_samples_split
min_samples_leaf
```

A total of **45 parameter combinations** were evaluated.

### Best Parameters

```text
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
```

### Best Cross-Validation Accuracy

**96.74%**

---

## Final Model Performance

The tuned Decision Tree achieved:

### Test Accuracy

**97.40%**

### Classification Report

```text
              precision    recall  f1-score   support

         acc       0.96      0.92      0.94        77
        good      0.88      1.00      0.93        14
       unacc      0.98      1.00      0.99       242
       vgood      1.00      0.85      0.92        13

    accuracy                           0.97       346
   macro avg       0.95      0.94      0.95       346
weighted avg       0.97      0.97      0.97       346
```

The **macro F1-score of 0.95** provides a better view of performance across the four classes because the dataset is imbalanced.

---

## Confusion Matrix

The final model produced the following confusion matrix:

```text
[[ 71   2   4   0]
 [  0  14   0   0]
 [  1   0 241   0]
 [  2   0   0  11]]
```

![Confusion Matrix](images/confusion_matrix.png)

The model correctly classified **337 out of 346** test samples.

Only **9 samples were misclassified**.

---

## Decision Tree Visualization

The complete Decision Tree contains many nodes because the best configuration did not impose a maximum depth.

For easier interpretation, the top three levels of the trained tree are visualized below:

![Decision Tree](images/decision_tree_top3.png)

This visualization helps demonstrate how the model creates decision rules from the encoded categorical features.

---

## Feature Importance

The Decision Tree provides feature importance scores that indicate which encoded features contributed most to the model's decisions.

### Top Important Features

| Feature | Importance |
|---|---:|
| `persons_2` | 0.2301 |
| `safety_low` | 0.1576 |
| `lug_boot_small` | 0.0824 |
| `maint_low` | 0.0758 |
| `safety_med` | 0.0639 |
| `maint_vhigh` | 0.0515 |
| `lug_boot_big` | 0.0477 |
| `maint_med` | 0.0463 |
| `safety_high` | 0.0415 |
| `doors_2` | 0.0411 |

![Feature Importance](images/feature_importance.png)

The results indicate that **passenger capacity, safety, luggage boot size, and maintenance-related features** played important roles in the learned decision rules.

---

## Model Saving

The complete preprocessing and Decision Tree pipeline was saved using Joblib:

```python
joblib.dump(
    best_dt_model,
    "decision_tree_car_evaluation.pkl"
)
```

The saved pipeline includes:

- One-Hot Encoder
- Decision Tree Classifier

The saved model was also loaded again and tested to verify that its predictions matched the original model.

```text
Predictions match: True
```

---

## Project Structure

```text
04_Decision_Tree/
└── Car_Evaluation_Classification/
    ├── Decision_Tree_Car_Evaluation.ipynb
    ├── README.md
    ├── requirements.txt
    ├── decision_tree_car_evaluation.pkl
    │
    └── images/
        ├── car_class_distribution.png
        ├── confusion_matrix.png
        ├── feature_importance.png
        └── decision_tree_top3.png
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab

---

## Key Machine Learning Concepts Demonstrated

- Categorical data preprocessing
- One-Hot Encoding
- Stratified train-test splitting
- Decision Tree Classification
- Hyperparameter tuning
- GridSearchCV
- Stratified K-Fold Cross-Validation
- Classification metrics
- Confusion matrix analysis
- Feature importance
- Model serialization
- Machine learning pipelines

---

## Limitations

- The dataset contains a significant class imbalance.
- The `good` and `vgood` classes contain relatively few samples.
- The test set contains only 346 samples.
- The reported accuracy is specific to this train-test split and should not be interpreted as guaranteed real-world performance.
- Decision Trees can become complex and may overfit when unrestricted.

---

## Future Improvements

Possible improvements include:

- Experimenting with Decision Tree pruning and depth constraints
- Comparing performance with Random Forest and Gradient Boosting
- Investigating class-weighted training
- Performing additional validation on external data
- Comparing multiple classification algorithms
- Exploring more robust model evaluation techniques

---

## Conclusion

This project demonstrates a complete **Decision Tree classification workflow** using categorical car evaluation data.

The final model achieved **97.40% test accuracy** and a **0.95 macro F1-score** while providing interpretable decision rules and feature importance information.

The project highlights the importance of proper categorical encoding, stratified evaluation, cross-validation, hyperparameter tuning, and model interpretability when building a practical machine learning classification system.
