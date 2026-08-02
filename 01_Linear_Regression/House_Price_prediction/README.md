# 🏠 House Price Prediction using Linear Regression

> An end-to-end Machine Learning project that predicts California house prices using Linear Regression. This project demonstrates the complete ML workflow including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and model saving.

---

## 📌 Project Overview

The objective of this project is to build a machine learning model capable of predicting house prices based on housing characteristics such as income, house age, average rooms, population, and geographical location.

This project follows a complete machine learning pipeline from data exploration to model evaluation.

---

## 🎯 Problem Statement

Accurately estimating house prices is important for:

- Real estate companies
- Home buyers
- Property investors
- Financial institutions

The goal is to predict the **median house value** using various housing features.

---

## 📂 Dataset

**Dataset:** California Housing Dataset

The dataset is provided by **Scikit-learn** and contains **20,640 housing records**.

### Features

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

### Target

- HouseValue

---

## 🛠️ Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 🔄 Machine Learning Workflow

```

Load Dataset
↓

Exploratory Data Analysis (EDA)
↓

Data Cleaning
↓

Data Visualization
↓

Data Preprocessing
↓

Train-Test Split
↓

Linear Regression Model
↓

Prediction
↓

Model Evaluation
↓

Model Saving

```

---

## 📊 Exploratory Data Analysis

The following analyses were performed:

- Dataset shape
- Data types
- Missing value analysis
- Summary statistics
- Correlation heatmap
- House price distribution

---

## 🤖 Model

Algorithm Used:

**Linear Regression**

The model was trained using:

- 80% Training Data
- 20% Testing Data

Random State:

```
42
```

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Mean Absolute Error (MAE) | **0.5332** |
| Mean Squared Error (MSE) | **0.5559** |
| Root Mean Squared Error (RMSE) | **0.7456** |
| R² Score | **0.5758** |

---

## 📷 Project Visualizations

### House Price Distribution

![House Price Distribution](images/distribution.png)

---

### Correlation Heatmap

![Correlation Heatmap](images/heatmap.png)

---

### Actual vs Predicted Values

![Actual vs Predicted](images/actual_vs_predicted.png)

---

## 💾 Model Saving

The trained model is saved using Joblib.

```python
joblib.dump(model, "linear_regression_model.pkl")
```

---

## 🚀 Future Improvements

- Feature Engineering
- Hyperparameter Optimization
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

---

## 📁 Project Structure

```

House_Price_Prediction/
│
├── House_Price_Prediction.ipynb
├── README.md
├── requirements.txt
├── linear_regression_model.pkl
├── images/
├── interview_questions.md

```

---

## 👨‍💻 Author

**Seela SaiRani**

Machine Learning Enthusiast

Building an end-to-end Machine Learning Portfolio one project at a time.

---

## ⭐ If you found this project useful, consider giving it a star!
