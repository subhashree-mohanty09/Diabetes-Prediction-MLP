# Diabetes Prediction using Multilayer Perceptron


## Live Demo

🚀 [Try the Diabetes Prediction App](https://diabetes-prediction-mlp-9kajci8m9g4cmhpd7trjez.streamlit.app/)

## Project Overview

This project focuses on predicting diabetes using the **Pima Indians Diabetes Dataset**. The project follows a complete machine learning pipeline, starting from data exploration and preprocessing to model training, evaluation, saving the final model, and deployment using Streamlit.

The main objective is to compare conventional machine learning models with a **Multilayer Perceptron (MLP)** and build a working diabetes prediction application.

## Dataset

The dataset contains **768 patient records** and **8 input features**:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The target variable is:

- `Outcome = 0` → Non-diabetic
- `Outcome = 1` → Diabetic

## Data Preprocessing and Feature Engineering

The project includes:

- Exploratory Data Analysis (EDA)
- Missing and suspicious value handling
- Outlier analysis
- Duplicate checking
- Median imputation for numerical features
- Categorical encoding
- Feature scaling
- Age groups
- BMI categories
- Glucose categories
- Interaction features
- Log transformation of Insulin
- Feature selection based on correlation and model importance

The data was split into **80% training and 20% testing** using stratification and a fixed random seed.

## Machine Learning Models

Five models were implemented and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. Multilayer Perceptron (MLP)

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73.38% | 63.27% | 57.41% | 60.19% | 83.87% |
| KNN | 76.62% | 68.75% | 61.11% | 64.71% | 84.12% |
| SVM | 74.03% | 64.58% | 57.41% | 60.78% | 80.85% |
| Random Forest | 75.32% | 62.90% | 72.22% | 67.24% | 82.31% |
| Tuned MLP | 80.0% | 71.70% | 70.37% | 71.03% | 84.96% |

The tuned MLP achieved the highest accuracy, F1-score, and ROC-AUC among the five evaluated models.

## MLP Architecture

The final MLP uses:

- Input layer
- Dense layer with 64 neurons and ReLU activation
- Dropout of 30%
- Dense layer with 32 neurons and ReLU activation
- Dropout of 20%
- Dense layer with 16 neurons and ReLU activation
- Output layer with sigmoid activation

The model uses the **Adam optimizer**, binary cross-entropy loss, batch size of 32, and early stopping.

## Streamlit Application

The final model is deployed using **Streamlit**.

The application accepts patient information such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

It then performs the required preprocessing and displays:

- Diabetes prediction
- Prediction probability

## Project Structure

```text
Diabetes Prediction/
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── final_mlp.keras
│   └── preprocessor.joblib
│
├── notebooks/
│   └── Diabetes_Prediction_MLP.ipynb
│
├── screenshots/
│
├── src/
│   └── app.py
│
├── Diabetes Prediction Report/
│
├── requirements.txt
├── README.md
└── .gitignore
