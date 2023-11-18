from typing import Literal
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(X, y, w):
    n = len(y)
    h = sigmoid(np.dot(X, w))
    J = (-1 / n) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return J


class NaiveBayes:
    """Naive Bayes class.

    If k is different than 0, will find the best alpha, tol and reg_cst
    using k-fold cross-validation.

    Attributes:
        thetas (ndarray): ndarray of shape (k, (1 + m))
          Theta values for classification
        class: Output class
        n_features (int): Dimension of features (m). 0 if model not fitted.
        n_iter (int): Number of iterations needed for fitting. 0 if not fitted.

        X (ndarray): Training inputs (nxm)
        y (ndarray): Training output (nx1)
    """

    def __init__(
        self,
        laplace_smoothing: bool = True,
        verbose=False,  # To print execution info
    ) -> None:
        self._n_features = 0

        self._classes = None
        self._n_class = 0
        self._class_count = None
        self._n_samples = 0

        self.X = None  # X dataset for training
        self.y = None  # y dataset for training

        self.laplace_smoothing = laplace_smoothing

        self._log_class_prior = None
        self._feat_log_proba = None
        self._feat_log_proba = None

        self.results = None  # Results dataframe
        self._comp_time = None

        self._verbose = verbose

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
            X (ndarray) : shape (n_samples, n_features)
                Training vector.

            y (ndarray) : shape (n_samples,)
                Expected output vector

            w (ndarray, optional): shape (n_features, )
            T   o give an initial guess

        Returns:
            self
                Model with weights fitted to training dataset
        """
        self.X = X
        self.y = y

        # Check dataset sizes
        self._n_samples, self._n_features = X.shape

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch between the size of the input ({X.shape[0]}) and outputs ({y.shape[0]})")

        # Number of class
        self._classes, self._class_count = np.unique(self.y, return_counts=True)
        self._n_class = len(self._classes)

        if self._verbose:
            print(f"Fitting Naive Bayes model for the dataset")
            print(f"\t# Features: {self._n_features}, classes: {self._classes}, # Samples: {self._n_samples}")

        self._train_model()

        return self

    def predict(self, X):
        """To predict the output of samples.

        For each class, computes:
            $$  \delta_k = \log{[\theta_k\sum_{j=1}^m \theta_{j, k}^{x_j} (1 - \theta_{j, k})^{1-x_j} ]}$$

        Classify the output as:
            $$ \text{Output} = \argmax_k \delta_k(x) $$

        Args:
            X (ndarray): Inputs sample to be predicted. Size (nxm)

        Raises:
            ValueError: If model is not trained

        Returns:
            ndarray: Predicted output of each data point (nx1)
        """
        if self._thetas is None:
            raise ValueError(f"Model is not trained.")

        log_class_prior = np.log(self._thetas[:, 0])  # log( P(y=k) )  shape: 1xk
        feat_log_proba = np.log(self._thetas[:, 1::])  # log( P(x_j=1 | y=k) )  shape: kxm
        feat_log_neg_proba = np.log(1 - self._thetas[:, 1::])  # log( 1 - P(x_j=1 | y=k) )  shape: kxm

        n_samples = X.shape[0]
        predictions = []
        self._joint_log_likelihood = np.zeros((n_samples, self._n_class))

        # # SK
        # from sklearn.utils.extmath import safe_sparse_dot

        # jll = safe_sparse_dot(X, (feat_log_proba - feat_log_neg_proba).T)
        # jll += log_class_prior + feat_log_neg_proba.sum(axis=1)

        # Compute discriminants of classes for all samples
        # Σ [ x_j * (log P(x|y) - log( 1 - P(x|y) ) + log( 1-P(x|y) )]
        self._joint_log_likelihood = X @ (feat_log_proba - feat_log_neg_proba).T + feat_log_neg_proba.sum(axis=1)

        # Add class log priors
        self._joint_log_likelihood += log_class_prior

        # Predictions
        predictions_idx = np.argmax(self._joint_log_likelihood, axis=1)
        predictions = self._classes[predictions_idx]

        # for sample_num in range(n_samples):
        #     X_pred = X[sample_num, :]

        #     feat_proba = np.sum(
        #         feat_log_proba * X_pred + feat_log_neg_proba * (1 - X_pred), axis=1
        #     )

        #     discriminants = log_class_prior + feat_proba

        #     self._joint_log_likelihood[sample_num, :] = discriminants

        #     y_pred = self._classes[np.argmax(discriminants, axis=None)]

        #     predictions.append(y_pred)

        return predictions

    def score(self, X, y):
        """To compute the accuracy of the model

        Args:
            X (ndarray): Test samples
            y (ndarray): True class of X

        Returns:
            float: Accuracy of the model over test samples
        """
        y_pred = self.predict(X)

        accuracy = (y == y_pred).mean()

        return accuracy

    def _train_model(self):
        """Compute the theta needed to estimate the probabilities

        For each class:
            $$ \theta_k = P(y=k) = (# samples where y=k) / (# samples) $$
            $$ \theta_{j, k} = P(x_j=1 | y=k) = (# samples where x_j=1 and y=k) / (# samples where y=k)  $$

        Stores the values in a np.array of shape k x (1 + m)
            -> \theta_{k} = thetas[k, 0], k=0, ... n_class  (prior of class k)
            -> \theta_{j, k} = thetas[k, j], j=1, ..., m
        """
        self._thetas = np.zeros([self._n_class, self._n_features + 1])

        for k, class_label in enumerate(self._classes):
            n_yk = self._class_count[k]  # n samples where y=k

            X_k = self.X[self.y == class_label, :]

            theta_k = n_yk / self._n_samples  # P(y=k), prior for class k
            self._thetas[k, 0] = theta_k

            for j in range(self._n_features):
                samples_j_k = X_k[:, j] == 1  # Array with True where X_j is 1

                n_xj_yk = samples_j_k.sum()  # n samples where y=k and x=x_j

                if self.laplace_smoothing:
                    theta_j_k = (n_xj_yk + 1) / (n_yk + 2)
                else:
                    theta_j_k = n_xj_yk / n_yk

                self._thetas[k, j + 1] = theta_j_k  # \theta_{j, k}


class MyMultinomialNB:
    def __init__(self):
        self.class_probabilities = {}
        self.word_probabilities = {}
        self.n_classes_ = 0
        self.classes_ = []
        self.jll_ = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Calculate class probabilities
        self.classes_, self.n_classes_ = np.unique(y, return_counts=True)
        self.class_probabilities = dict(zip(self.classes_, self.n_classes_ / num_samples))

        # Calculate word probabilities for each class
        for cls in self.classes_:
            cls_indices = y == cls
            class_word_count = X[cls_indices].sum(axis=0)  # Total each feature appears in for class k
            total_word_count = X[cls_indices].sum() + num_features  # Laplace smoothing
            self.word_probabilities[cls] = (class_word_count + 1) / total_word_count

        return self

    def predict(self, X):
        n_samples, n_features = X.shape
        num_classes = len(self.classes_)
        self.jll_ = np.zeros([n_samples, num_classes])

        for i in range(X.shape[0]):
            jll_i = np.zeros(num_classes)
            X_i = X[i] if isinstance(X[i], np.ndarray) else X[i].toarray()
            X_i = X_i.reshape(1, -1)
            for k, (cls, cls_prob) in enumerate(self.class_probabilities.items()):
                word_prob = self.word_probabilities[cls].reshape(1, -1)

                jll_i[k] = np.log(cls_prob) + np.dot(X_i, np.log(word_prob).T)

            self.jll_[i, :] = jll_i

        indices = np.argmax(self.jll_, axis=1)
        return self.classes_[indices]

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
