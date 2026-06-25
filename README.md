# Credit Card Fraud Detection Using ANN

A PyTorch-based Artificial Neural Network (ANN) for detecting fraudulent credit card transactions. The project includes data preprocessing, feature scaling, SMOTE for class balancing, model training, and performance evaluation.

## Features

* Data preprocessing and feature encoding
* SMOTE for handling class imbalance
* ANN with Batch Normalization and Dropout
* Performance evaluation using Accuracy, ROC-AUC, and Confusion Matrix

## Tech Stack

Python, PyTorch, Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn

## Dataset

The dataset is too large to be included in this repository.

Download it from Kaggle:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Place the following files in the project directory before running the project:

* `fraudTrain.csv`
* `fraudTest.csv`

## Run

```bash
pip install pandas numpy torch scikit-learn imbalanced-learn matplotlib seaborn
python fraud_detection.py
```
