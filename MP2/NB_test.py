from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

import numpy as np
import matplotlib.pyplot as plt

from NaiveBayes import NaiveBayes, MyMultinomialNB

import warnings

from sklearn.feature_extraction.text import CountVectorizer


# --- Datasets ---
# Iris dataset
# test_size = 0.2
# X, y = load_iris(return_X_y=True)
# print('iris', X.shape, y.shape)
# print("number of classes is ", len(np.unique(y)))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(f"Trainin dataset size: {X_train.shape}")

# Binary Features toy dataset
rng = np.random.RandomState(2)
X = rng.randint(2, size=(9, 5))
# y = np.array([1, 2, 3, 3, 1, 2, 3, 1, 2])
y = np.array(["a", "b", "c", "c", "a", "b", "c", "a", "b"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# --- SK Learn Naive Bayes ---
sk_nb = BernoulliNB()
sk_nb.fit(X_train, y_train)
y_pred_sk = sk_nb.predict(X_test)

print(f"SK Learn Bernouilli NB Results:")
attributes = [
    "class_count_",
    "class_log_prior_",
    "classes_",
    "feature_count_",
    "feature_log_prob_",
    "n_features_in_",
]
for att in attributes:
    print(f'\t{att}:\n\t\t {eval(f"sk_nb.{att}")}')

print(f"Test Joint Log Likelihood: \n{sk_nb.predict_joint_log_proba(X_test)}\n")

# --- SK Learn Multinomial Naive Bayes ---
sk_mn_nb = MultinomialNB()
sk_mn_nb.fit(X_train, y_train)
y_pred_sk_mn_nb = sk_mn_nb.predict(X_test)

# print(f"SK Learn Bernouilli NB Results:")
# attributes = [
#     "class_count_",
#     "class_log_prior_",
#     "classes_",
#     "feature_count_",
#     "feature_log_prob_",
#     "n_features_in_",
# ]
# for att in attributes:
#     print(f'\t{att}:\n\t\t {eval(f"sk_mn_nb.{att}")}')
#
# print(f"Test Joint Log Likelihood: \n{sk_mn_nb.predict_joint_log_proba(X_test)}\n")

# --- My Class ---
my_nb = NaiveBayes(laplace_smoothing=True, verbose=True)

my_nb.fit(X_train, y_train)

y_pred_my = my_nb.predict(X_test)


# --- My Multinomial NB ---
my_mn_nb = MyMultinomialNB()

my_mn_nb.fit(X_train, y_train)

y_pred_my_mn_nb = my_mn_nb.predict(X_test)


# # --- Compare Implementation Bernoulli ---
# print(f"\n\n--- Comparing Results ---")
# print(f"Class Log Prior:")
# print(f"SK: \t\t{sk_nb.class_log_prior_}")
# print(f"My NB: \t\t{np.log(my_nb._thetas[:, 0])}")
# try:
#     assert np.allclose(sk_nb.class_log_prior_, np.log(my_nb._thetas[:, 0]))
# except:
#     warnings.warn("Mismatch between class log prior")
#
# print(f"\nFeatures log proba:")
# for k, label in enumerate(my_nb._classes):
#     print(f"\tClass: {label}")
#     print(f"\tSK: \t\t{sk_nb.feature_log_prob_[k, :]}")
#     print(f"\tMy NB: \t\t{np.log(my_nb._thetas[k, 1::])}\n")
#     try:
#         assert np.allclose(sk_nb.feature_log_prob_[k, :], np.log(my_nb._thetas[k, 1::]))
#     except:
#         warnings.warn(f"Mismatch between feature log proba for label {label} (idx: {k})")
#
#
# print(f"\nPredictions")
# print(f"\tSK: \t{y_pred_sk}")
# print(f"\tMy NB: \t{y_pred_my}")
# try:
#     assert (y_pred_sk == y_pred_my).all()
# except:
#     warnings.warn("Mismatch between predictions")
#
# print(f"\nJoint Log Likelihood")
# for i in range(X_test.shape[0]):
#     sk_val = sk_nb.predict_joint_log_proba(X_test)[i, :]
#     my_val = my_nb._joint_log_likelihood[i, :]
#     print(f"\tSK: \t{sk_val}")
#     print(f"\tMy NB: \t{my_val}\n")
#     try:
#         assert np.allclose(sk_val, my_val)
#     except:
#         warnings.warn(f"Mismatch between joint log likelihood for sample {i}\n\n")

# --- Compare Implementation Multinomial ---
print(f"\n\n--- Comparing Results ---")
print(f"Class Log Prior:")
print(f"SK: \t\t{sk_mn_nb.class_log_prior_}")
my_nb_priors = np.fromiter(my_mn_nb.class_probabilities.values(), dtype=float)
print(f"My NB: \t\t{np.log(my_nb_priors)}")
try:
    assert np.allclose(sk_mn_nb.class_log_prior_, np.log(my_nb_priors))
except:
    warnings.warn("Mismatch between class log prior")

print(f"\nFeatures log proba:")
for k, label in enumerate(my_mn_nb.classes_):
    print(f"\tClass: {label}")
    print(f"\tSK: \t\t{sk_mn_nb.feature_log_prob_[k, :]}")
    print(f"\tMy NB: \t\t{np.log(my_mn_nb.word_probabilities[label])}\n")
    try:
        assert np.allclose(sk_mn_nb.feature_log_prob_[k, :], np.log(my_mn_nb.word_probabilities[label]))
    except:
        warnings.warn(f"Mismatch between feature log proba for label {label} (idx: {k})")

print(f"\nPredictions")
print(f"\tSK: \t{y_pred_sk_mn_nb}")
print(f"\tMy NB: \t{y_pred_my_mn_nb}")
try:
    assert (y_pred_sk_mn_nb == y_pred_my_mn_nb).all()
except:
    warnings.warn("Mismatch between predictions")

print(f"\nJoint Log Likelihood")
for i in range(X_test.shape[0]):
    sk_val = sk_mn_nb.predict_joint_log_proba(X_test)[i, :]
    my_val = my_mn_nb.jll_[i, :]
    print(f"\tSK: \t{sk_val}")
    print(f"\tMy NB: \t{my_val}\n")
    try:
        assert np.allclose(sk_val, my_val)
    except:
        warnings.warn(f"Mismatch between joint log likelihood for sample {i}\n\n")

...
