# Employee Attrition Prediction System

## Overview

This project predicts whether an employee is likely to leave an organization using Machine Learning and Ensemble Learning techniques.

The system uses a Stacked Ensemble Classifier consisting of:

* Random Forest
* XGBoost
* Logistic Regression (Meta Learner)

The model was trained on a combined HR dataset containing 11,058 employee records and achieved high predictive performance.

---

## Live Demo

Streamlit App:
[Your Streamlit URL]

---

## Features

* Employee Attrition Prediction
* Interactive Web Interface
* Machine Learning Pipeline
* Ensemble Learning Model
* HR Analytics Dashboard
* Real-Time Prediction

---

## Dataset

Combined Dataset Size: 11,058 Records

Features Include:

* Age
* Monthly Income
* Job Satisfaction
* Environment Satisfaction
* Work Life Balance
* Distance From Home
* Years At Company
* Job Role
* Department
* Overtime Status
* Marital Status

---

## Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. One-Hot Encoding
5. Min-Max Scaling
6. PCA Dimensionality Reduction
7. Model Training
8. Ensemble Stacking
9. Deployment using Streamlit

---

## Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 98.23% |
| Precision | 99.65% |
| Recall    | 96.79% |
| F1 Score  | 98.20% |
| AUC ROC   | 98.74% |

---

## Technologies Used

* Python
* Scikit-Learn
* XGBoost
* Pandas
* NumPy
* Streamlit
* GitHub

---

## Project Structure

employee-attrition/

├── app.py

├── best_attrition_model.pkl

├── final_clean_dataset.csv

├── requirements.txt

├── docs/

│ ├── Employee_Attrition_Paper.pdf

│ └── Employee_Attrition_Report.pdf

└── README.md

---

## Research Contributions

* Combined dataset of 11,058 employee records
* PCA based dimensionality reduction
* Class imbalance handling
* Stacked Ensemble Learning
* Deployment as a live web application

---

## Authors

Tanish Shah 
