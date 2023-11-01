from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import numpy as np
import matplotlib.pyplot as plt

# Load datasets
test_size = 0.2

X, y = load_iris(return_X_y=True)


print('iris', X.shape, y.shape)
print("number of classes is ", len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Trainin dataset size: {X_train.shape}")

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

...
print(f'--- Fitting results ---')
for idx, each_class in enumerate(gnb.classes_):
    print(f'Class: {each_class}')
    print(f'Prior: {gnb.class_prior_[idx]}')
    print(f'Thetas: {gnb.var_[idx, :]}\n')