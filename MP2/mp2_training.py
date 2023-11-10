"""
To find the best model and their parameter combination using K-Fold validation
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from NaiveBayes import NaiveBayes
from cross_val_score import cross_val_score

#TODO
# - Define list of dataset for CV
# - Analyze results df
# - Use text datasets
# - Define more models

# Define training datasets
# Binary Features toy dataset
rng = np.random.RandomState(1)
X = rng.randint(2, size=(9, 5))
# y = np.array([1, 2, 3, 3, 1, 2, 3, 1, 2])
y = np.array(["a", "b", "c", "c", "a", "b", "c", "a", "b"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


class Dataset:
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.name = 'dataset1'


##
train_data_1 = Dataset(X_train, y_train)


##
model_list = {}
model_list["My NB"] = {
    "model": NaiveBayes,
    "train_data": train_data_1,
    'base_params': {'laplace_smoothing': True, 'k_cv': 0, 'verbose': False},
    'cv_params': None
}

model_list["SK Bernoulli NB"] = {
    "model": BernoulliNB,
    "train_data": train_data_1,
    'base_params': {},
    'cv_params': None
}

model_list["SVM"] = {
    "model": svm.LinearSVC,
    "train_data": train_data_1,
    "base_params": {"random_state": 0},
    "cv_params": {"C": [0.1, 1, 10], "penalty": ["l1", "l2"], "tol": {1e-4, 1e-5}},
}

# Cross-Validation
n_fold = 5
results_df = pd.DataFrame()

start_time = time.time()
print(f"--------- Training all models ---------")
for model_name, model_info in model_list.items():
    model = model_info["model"]
    model_train_data = model_info["train_data"]
    base_params = model_info["base_params"]
    cv_params = model_info["cv_params"]
    dataset_name = model_train_data.name

    X_train = model_train_data.X
    y_train = model_train_data.y

    print(f"\nModel : {model_name}")

    # Cross_validation
    cv_results = cross_val_score(
        model, X_train, y_train, cv=n_fold, base_params=base_params, cv_params=cv_params
    )

    best_row = cv_results.iloc[cv_results['Score'].idxmax()]
    print(f"\tBest Score : {best_row['Score']} with params: {best_row['Params']}")

    cv_results['Model'] = model_name
    cv_results['Dataset'] = dataset_name

    results_df = pd.concat([results_df, cv_results], ignore_index=True)

print(f"\nTraining completed ({time.time() - start_time} sec)")
print(results_df)
