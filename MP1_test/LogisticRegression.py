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


def gradiant_descent(
    X,
    y,
    start,
    alpha,
    tolerance,
    max_iter,
    reg: Literal["l2"] | None = None,
    reg_cst=1,
    lr_type="normal",
    verbose=False,
):
    """Gradient descent algo.

    Args:
        X (ndarray): Training data samples (nx,)
        y (ndarray): Sample output (nx1)
        start (ndarray): Weights initialization (mx1)
        alpha (float): Learning rate
        tolerance (float): Stop tolerance
        max_iter (int): Max # of iterations.
        reg (str, optional): Type of regularization to use. None, l2 or l1. Defaults to None.
        c_reg (float): Regularization term cst

    Returns:
        w : Weights found
        i : Number of iteration taken
        weights : List with weights over all iteration
        costs : List with cost function over all iteration
    """
    if verbose:
        print(f"\n--- Gradiant Descent ---")

    n_data, n_features = X.shape
    converged = False

    if start is None:
        w = np.ones(n_features)
    else:
        w = start.copy()

    w_tol_array = []
    costs = [cost_function(X, y, w)]
    weights = [w]

    if verbose:
        print(f"Starting weights: {w}")
        print(f"Learning rate: {alpha}\t Type: {lr_type}")
        print(f"Stop tolerance: {tolerance}")
        if reg is not None:
            print(f"Regulariztion: Type: {reg.upper()}, Constant: {reg_cst}")
        else:
            print(f"Regulariztion: Type: None")
        print(f"Finding solution...")

    start_time = time.time()
    for i in range(max_iter):
        w_prev = w.copy()

        # Regularization
        reg_term = 0
        if reg == "l2":
            reg_term = (
                2 * reg_cst / n_data * (np.insert(w[1:], 0, 0))
            )  # Insert 0 at beginning, No weight on bias term

        elif reg == "l1":
            reg_term = (
                reg_cst / n_data * (np.insert(np.sign(w[1:]), 0, 0))
            )  # Insert 0 at beginning, No weight on bias term

        elif reg is None:
            reg_term = 0

        else:
            raise ValueError(f"Invalid regularization type: {reg}")

        # Gradient
        est = sigmoid(np.dot(X, w))
        err = est - y
        grad = (1 / n_data) * np.dot(X.T, err)

        # Learning rate
        if lr_type == "normal":
            lr = alpha
        elif lr_type == "decaying":
            lr = alpha / (1 + i)
        else:
            raise ValueError(f"Invalid learning rate type: {lr_type}")

        # Update weights
        w -= lr * (grad + reg_term)

        weights.append(w.copy())

        # Cost
        cost = cost_function(X, y, w)
        costs.append(cost)

        # Check tolerance
        if i > 0:
            w_change = w - w_prev
            w_tol = np.dot(w_change, w_change)
            w_tol_array.append(w_tol)

            if w_tol < tolerance:
                converged = True
                break

    comp_time = (time.time() - start_time) * 1000

    if converged:
        if verbose:
            print(
                f"Solution found after {i} iterations ({comp_time} ms). Optimal cost: {cost}"
            )
    else:
        if verbose:
            print(
                f"Max Step reached ({i}) ({comp_time} ms) - Unable to find solution within tolerance ({tolerance})."
            )
            print(f"Final cost: {cost}")

    return w, i, weights, costs, converged, comp_time


class LogisticRegression:
    """Logistic Regression class.

    If k is different than 0, will find the best alpha, tol and reg_cst
    using k-fold cross-validation.

    The learning rate type can be change between normal and decaying.
        - In normal the rate is cst to alpha
        - In decaying the rate is : alpha/(1+k) where k is the epoch

    Attributes:
        weights (ndarray): ndarray of shape (1, n_features)
          Coefficient of the features. None if model not fitted.
        n_features (int): Dimension of features (m). 0 if model not fitted.
        n_iter (int): Number of iterations needed for fitting. 0 if not fitted.

        X (ndarray): Training inputs (nxm)
        y (ndarray): Training output (nx1)
    """

    def __init__(
        self,
        alpha: list
        | float = 0.01,  # Gradiant descent learning rate. List of values for CV.
        learning_rate_type: Literal[
            "normal", "decaying"
        ] = "normal",  # Type of learning rate
        tol: list | float = 1e-4,  # Stopping criteria tolerance. List of values for CV.
        max_iter: int = 100,  # Max number of iteration of the solver
        reg: Literal["l2"] | None = "l2",  # Form of regularization
        reg_cst: list
        | float = 1.0,  # Regularization constant. Higher means stronger reg. List of values for CV.
        k_cv: int = 0,  # k-fold cross-validation
    ) -> None:
        self.reg = reg
        self.tol = tol
        self.reg_cst = reg_cst
        self.max_iter = max_iter
        self.alpha = alpha
        self.lr_type = learning_rate_type

        self.weights = None
        self.n_features = 0
        self.n_iter = 0

        self.X_train = None
        self.y_train = None

        self.k_cv = k_cv

        self._weight_array = []  # Array with progression of weights during training
        self._cost_array = []  # Array with progression of cost during training

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
              To give an initial guess

        Returns
          self
              Model with weights fitted
        """
        self.X_train = X
        self.y_train = y

        self.n_features = X.shape[1]

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch between the size of the input ({X.shape[0]}) and outputs ({y.shape[0]})"
            )

        # No cross-validation
        if self.k_cv == 0:
            # Gradiant Descent Algorithm
            (
                self.weights,
                self.n_iter,
                self._weight_array,
                self._cost_array,
                converged,
                self._comp_time,
            ) = gradiant_descent(
                X,
                y,
                w_start,
                self.alpha,
                self.tol,
                self.max_iter,
                reg=self.reg,
                reg_cst=self.reg_cst,
                verbose=verbose,
                lr_type=self.lr_type,
            )

        # k-fold cross-validation
        else:
            self.cross_validation(X, y, w_start, verbose)

        return self

    def predict(self, X):
        """To predict the output of samples

        Args:
            X (ndarray): Inputs sample to be predicted. Size (nxm)

        Raises:
            ValueError: If model is not trained

        Returns:
            ndarray: Predicted output of each data point (nx1)
        """
        if self.weights is None:
            raise ValueError(f"Model is not trained.")

        p_class_1 = sigmoid(np.dot(X, self.weights))

        y_pred = (p_class_1 >= 0.5).astype(int)

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

    def plot_training(self, plot_cost=True, plot_weights=True):
        """Plot the progression of the weight during the training.
        Only works for a 1D or 2D feature space.

        Args:
            plot_cost (bool, optional): To plot the evolution of the cost function. Defaults to True.
            plot_weights (bool, optional): To plot the evolution of the weights. Defaults to True.
        """
        if plot_cost:
            # Cost progression
            plt.plot(self._cost_array)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Cost Progression Over Iteration")
            plt.show(block=False)

        if plot_weights:
            if self.weights.shape[0] > 2:
                print(f"ERROR: Dimension of feature space too large for plotting")
                return

            if self.weights.shape[0] == 2:
                # Generate 2D grid
                plot_range = 5
                n_points = 100
                w_0 = np.linspace(self.weights[0] - 3, self.weights[0] + 1, n_points)
                w_1 = np.linspace(
                    self.weights[1] - plot_range, self.weights[1] + 1, n_points
                )
                w_00, w_11 = np.meshgrid(w_0, w_1)
                w_grid = np.array([w_00.flatten(), w_11.flatten()])

                # Compute cost function values (Z)
                J_vals = cost_function(self.X_train, self.y_train, w_grid)
                J_vals = J_vals.reshape(w_00.shape)

                # Max value of J
                min_index = np.unravel_index(np.argmin(J_vals, axis=None), J_vals.shape)
                J_min = J_vals[min_index]

                w_0_min = w_0[min_index[1]]
                w_1_max = w_1[min_index[0]]

                cost_function(self.X_train, self.y_train, np.array([w_0_min, w_1_max]))
                print(f"Supposed Max of J: {J_min} for w_0: {w_0_min}, w_1: {w_1_max}")

                # Plot
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.contour(w_00, w_11, J_vals, 25)  # Contour of cost function

                weights_plot = np.array(self._weight_array)
                ax.scatter(weights_plot[:, 0], weights_plot[:, 1])
                ax.plot(weights_plot[:, 0], weights_plot[:, 1])

                ax.scatter(
                    w_0_min,
                    w_1_max,
                    color="green",
                    marker="x",
                    label="Absolute minimum",
                )
                plt.legend(loc="upper left")
                plt.xlabel("w_0")
                plt.ylabel("w_1")

                plt.show(block=False)

            elif self.weights.shape[0] == 1:
                # Generate 2D grid
                plot_range = 10
                n_points = 100

                w_0 = np.linspace(-7.5, 12.5, n_points).reshape(1, n_points)

                # Compute cost function values (Z)
                J_vals = cost_function(self.X_train, self.y_train, w_0)

                # Max value of J
                min_index = np.unravel_index(np.argmin(J_vals, axis=None), J_vals.shape)
                J_min = J_vals[min_index]

                w_0_min = w_0[0, min_index]

                cost_function(self.X_train, self.y_train, np.array([w_0_min]))
                print(f"Supposed Min of J: {J_min} for w_0: {w_0_min}")

                # Plot
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.plot(w_0[0, :], J_vals)  # Contour of cost function

                weights_plot = np.array(self._weight_array)
                ax.scatter(weights_plot[:, 0], self._cost_array)
                ax.plot(weights_plot[:, 0], self._cost_array)

                ax.scatter(
                    self.weights[0],
                    self._cost_array[-1],
                    color="blue",
                    marker="o",
                    label="Solution",
                )
                ax.scatter(
                    w_0_min,
                    J_min,
                    color="green",
                    marker="x",
                    label="Min",
                )
                plt.legend()
                plt.xlabel("w_0")
                plt.ylabel("Cost")

                plt.show(block=False)

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
