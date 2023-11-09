"""
File to find the best model
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from NaiveBayes import NaiveBayes


# Define training datasets
# Binary Features toy dataset
rng = np.random.RandomState(1)
X = rng.randint(2, size=(9, 5))
# y = np.array([1, 2, 3, 3, 1, 2, 3, 1, 2])
y = np.array(["a", "b", "c", "c", "a", "b", "c", "a", "b"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


class Dataset:
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train


##
train_data_1 = Dataset(X_train, y_train)


##
model_list = {}
model_list["My NB"] = {
    "model": NaiveBayes(laplace_smoothing=True, k_cv=0, verbose=False),
    "train_data": train_data_1,
}

model_list["SK Bernoulli NB"] = {
    "model": BernoulliNB(),
    "train_data": train_data_1,
}

# Cross-Validation
n_fold = 5
n_jobs = 1
results = []

start_time = time.time()
print(f"--------- Training all models ---------")
for model_name, model_info in model_list.items():
    model = model_info["model"]
    model_train_data = model_info["train_data"]
    params = ...

    X_train = model_train_data.X
    y_train = model_train_data.y

    print(f"\nModel : {model_name}")

    # Cross_validation
    cv_score = cross_val_score(model, X_train, y_train, cv=n_fold, n_jobs=n_jobs)
    print(f"\tScore : {cv_score}")

    results.append(
        {
            "Model": model,
            "Score": cv_score,
            "Parameters": params,
        }
    )

results = pd.DataFrame(results)
print(f"Training completed ({time.time() - start_time})")

# Print results
