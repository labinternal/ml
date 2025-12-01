# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 22:09:18 2025

@author: kommu
"""


# full, correct, ready-to-run script for your logistic regression example
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Column names (same as you used)
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin',
             'bmi', 'pedigree', 'age', 'label']

# Load dataset
# Use a raw string for Windows path to avoid escape issues.
csv_path = r"C:\Users\kommu\Downloads\diabetes.csv"
pima = pd.read_csv(csv_path, header=None, names=col_names)

# Print the full dataframe (as you did)
print(pima)

# Select features and label (same selection you provided)
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp']
X = pima[feature_cols]  # Features
y = pima.label          # Target variable

# Print X and y (as you did)
print(X)
print(y)

# Split into train and test sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Instantiate logistic regression
# Increased max_iter to avoid convergence warnings on some datasets
logreg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=200)

# Fit model
logreg.fit(X_train, y_train)

# Predict on test set
y_pred = logreg.predict(X_test)

# Confusion matrix and scores
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
