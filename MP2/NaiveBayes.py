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
        labels: Output labels
        n_features (int): Dimension of features (m). 0 if model not fitted.
        n_iter (int): Number of iterations needed for fitting. 0 if not fitted.

        X (ndarray): Training inputs (nxm)
        y (ndarray): Training output (nx1)
    """

    def __init__(
        self,
        laplace_smoothing: bool = True,
        k_cv: int = 0,  # k-fold cross-validation
    ) -> None:

        self.weights = None
        self.n_features = 0
        self.n_labels = 0
        self.n_iter = 0

        self.X_train = None
        self.y_train = None

        self.laplace_smoothing = laplace_smoothing

        self.k_cv = k_cv

        self._weight_array = []  # Array with progression of weights during training

        self.results = None  # Results dataframe
        self._comp_time = None

    def fit(self, X, y, w_start=None, verbose=False):
        """
        Fit the model according to the given training data.

        Parameters:
            X (ndarray) : shape (n_samples, n_features)
                Training vector.

            y (ndarray) : shape (n_samples,)
                Expected output vector

            w (ndarray, optional): shape (n_features, )
            T   o give an initial guess

            verbose (bool):
                To print info

        Returns:
            self
                Model with weights fitted to training dataset
        """
        self.X_train = X
        self.y_train = y

        n, self.n_features = X.shape

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch between the size of the input ({X.shape[0]}) and outputs ({y.shape[0]})"
            )
        
        # Number of labels
        self.n_labels = ...

        if verbose:
            print(f"Fitting Naive Bayes model for the dataset")
            print(f"\t# Features: {self.n_features}, # Labels: {self.n_labels}, # Samples: {n}")
            if self.k_cv != 0:
                print(f"\tUsing {self.k_cv}-Fold Cross-Validation")
            else:
                print(f"\tNo cross validation")

        # No cross-validation
        if self.k_cv == 0:
            self.train_model()

        # k-fold cross-validation
        else:
            self.cross_validation(X, y, w_start, verbose)

        return self

    # TODO
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
        if self.weights is None:
            raise ValueError(f"Model is not trained.")

        y_pred = ...

        return y_pred

    def accu_eval(self, X, y):
        """To compute the accuracy of the model

        Args:
            X (ndarray): Test samples
            y (ndarray): True labels of X

        Returns:
            float: Accuracy of the model over test samples
        """
        y_pred = self.predict(X)

        accuracy = (y == y_pred).mean()

        return accuracy

    # TODO
    def cross_validation(self, X, y, w_start, verbose=False):
        """Perform k-fold cross validation for a given model (set of parameters)/

        Args:
            k (int): Number of folds
            parameters (...): Model params

        Returns:
            ...: Accuracy, comp_time, model parameters
        """
        # 1. Determine combinations of parameters
        results = []  # To store the result of each combination

        alpha_array = np.array(self.alpha)  # Size i
        tol_array = np.array(self.tol)  # Size j
        reg_cst_array = np.array(self.reg_cst)  # Size k

        xx, yy, zz = np.meshgrid(alpha_array, tol_array, reg_cst_array)

        # Creates a 3x(i*j*k) array of all combinations.
        # The rows are [alpha;tol;lambda] and colums correspond to a combination
        param_combinations = np.array([xx.flatten(), yy.flatten(), zz.flatten()])

        # Fold sizes
        n_samples = X.shape[0]

        # Array with each fold size
        fold_sizes = np.full(self.k_cv, n_samples // self.k_cv, dtype=int)

        # Add remainder over the first indices
        fold_sizes[: n_samples % self.k_cv] += 1

        # 2. For each combination
        for each_param_comb in param_combinations.T:
            alpha = each_param_comb[0]
            tol = each_param_comb[1]
            reg_cst = each_param_comb[2]

            # For each fold
            acc_avg = 0
            comp_time_avg = 0
            curr_idx = 0

            for i in range(self.k_cv):
                # Split datasets
                start, end = curr_idx, curr_idx + fold_sizes[i]
                curr_idx = end

                X_val, y_val = X[start:end], y[start:end]
                X_train, y_train = np.concatenate((X[:start], X[end:])), np.concatenate(
                    (y[:start], y[end:])
                )

                # Fit model
                (
                    weights,
                    n_iter,
                    _weight_array,
                    _cost_array,
                    converged,
                    comp_time,
                ) = gradiant_descent(
                    X_train,
                    y_train,
                    w_start,
                    alpha,
                    tol,
                    self.max_iter,
                    reg=self.reg,
                    reg_cst=reg_cst,
                    lr_type=self.lr_type,
                    verbose=verbose,
                )

                self.weights = weights

                # Accuracy
                acc = self.accu_eval(X_val, y_val)

                acc_avg += acc
                comp_time_avg += comp_time

            acc_avg /= self.k_cv
            comp_time_avg /= self.k_cv

            # Add results to dataframe
            results.append(
                {
                    "Accuracy": acc_avg,
                    "Alpha": alpha,
                    "Tolerance": tol,
                    "Lambda": reg_cst,
                    "Comp time": comp_time_avg,
                    "Converged": converged,
                }
            )

        # 3. Print results and select best model
        results = pd.DataFrame(results)

        max_acc_rows = results[results["Accuracy"] == results["Accuracy"].max()]
        max_acc_row = max_acc_rows.iloc[0]

        acc_max = max_acc_row["Accuracy"]
        alpha_max = max_acc_row["Alpha"]
        tol_max = max_acc_row["Tolerance"]
        reg_cst_max = max_acc_row["Lambda"]
        converged = max_acc_row["Converged"]

        # Train model on whole dataset with best params
        if verbose:
            print(f"Max accuracy obtained: {acc_max}")
            print(f"\tAlpha: {alpha_max}")
            print(f"\tTolerance: {tol_max}")
            print(f"\tLambda: {reg_cst_max}")
            print(f"\tConverged: {converged}")

        (
            best_weights,
            n_iter,
            _weight_array,
            _cost_array,
            converged,
            comp_time,
        ) = gradiant_descent(
            X_train,
            y_train,
            w_start,
            alpha_max,
            tol_max,
            self.max_iter,
            reg=self.reg,
            reg_cst=reg_cst_max,
            lr_type=self.lr_type,
        )
        self.weights = best_weights

        self.results = results

        return results  # acc, comp time, parameters

    def train_model(self):
        """Compute the theta needed to estimate the probabilities

        For each class:
            $$ \theta_k = P(y=k) = (# samples where y=k) / (# samples) $$
            $$ \theta_{j, k} = P(x_j=1 | y=k) = (# samples where x_j=1 and y=k) / (# samples where y=k)  $$ 

        Stores the values in a np.array of shape k x (1 + m)
            -> \theta_{k} = thetas[k, 0], k=0, ... n_labels
            -> \theta_{j, k} = thetas[k, j], j=1, ..., m

        """
        self.thetas = np.zeros(self.n_labels, (self.n_features + 1))

        for k in range(self.n_labels):
            n_total = ... # n total of samples

            n_y_k = ... # n samples where y=k

            theta_k = ... # \theta_k

            self.thetas[k, 0] = theta_k

            for j in range(1, self.n_features):
                n_x_k = ... # n samples where y=k and x=x_j

                theta_j_k = ... # \theta_{j, k} 

                self.thetas[k, j] = theta_j_k


            if self.laplace_smoothing:
                ...