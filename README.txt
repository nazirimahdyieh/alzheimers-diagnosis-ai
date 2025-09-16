Alzheimer's Disease Classification

This repository contains Python code for classifying Alzheimer's disease using various machine learning models. The dataset used is from Kaggle: Alzheimer's Disease Dataset
.

The workflow includes data preprocessing, training multiple models, hyperparameter tuning, evaluation, and stacking ensemble methods.

Dataset

The dataset contains patient information and diagnostic labels (Diagnosis column) indicating the presence of Alzheimer's disease.

Features include medical test results and demographic information.

Dataset Link:

Some columns like PatientID and DoctorInCharge were excluded from modeling.
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset?utm_source=chatgpt.com

Preprocessing

Label encoding for categorical features.

Standardization of numerical features.

Splitting the dataset into training (80%) and testing (20%) sets.

Optional handling of target encoding if it is categorical.

Models Implemented
1. CatBoost Classifier

Used optimized hyperparameters:

iterations=302

depth=5

learning_rate=0.062

l2_leaf_reg=9.986

Achieved accuracy ≈ 95.9% on test data.

CatBoost handles categorical variables efficiently.

2. LightGBM & XGBoost

LightGBM and XGBoost models trained with default hyperparameters.

Accuracy for LightGBM ≈ 95.5%.

Accuracy for XGBoost ≈ 95.3%.

Both models are gradient boosting-based and are good for tabular data.

3. TabNet Classifier

Deep learning approach for tabular data using attention-based feature selection.

Optimized parameters:

n_d = n_a = 32

n_steps = 5

gamma = 1.5

Achieved competitive accuracy and handled complex feature interactions.

4. Stacking Ensemble

Combined CatBoost, LightGBM, and RandomForest as base models.

Logistic Regression as final estimator.

Achieved high accuracy (~96%) by leveraging strengths of multiple models.

Evaluation Metrics

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

Example (CatBoost):

Class	Precision	Recall	F1-score
0	0.96	0.96	0.96
1	0.93	0.92	0.93

Overall Accuracy: 0.959

How to Run

Clone this repository:

git clone <repository-url>


Install required packages:

pip install -r requirements.txt


Place the dataset CSV file in the data/ folder or update the path in the scripts.

Run individual scripts or a combined notebook for all models.