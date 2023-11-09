import numpy as np
from sklearn.model_selection import KFold
import itertools


def cross_val_score(model_class, X, y, cv=5, base_params={}, cv_params={}):
    """To perform K-Fold validation to find the best combination of parameters for a given model.

    The parameters in `base_params` are kept the same for all tests.

    K-fold validation is performed for each combination of params in `model_params`.

    Args:
        model_class (model class): Model to test
        X (NDArray): Training dataset
        y (NDArray): Labels of the training dataset
        cv (int, optional): Number of folds. Defaults to 5.
        base_params (dict, optional): Keywords to pass to the model. Defaults to {}.
        model_params (dict, optional): Keywords to pass to the model. Defaults to {}.

    Returns:
        TBD: TBD, all possible combinations and their score? best model and its score? best params?
    """
    kf = KFold(n_splits=cv, shuffle=False)

    scores = np.zeros(cv)

    # Find all combinations of parameters
    keys, values = zip(*cv_params.items())
    all_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Find the best combination w/ CV
    for each_comb in all_combs:
        model = model_class(**base_params, **each_comb)
        print(f"CV for params: {each_comb}")

        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # score = model.fit(X_train, y_train).score(X_test, y_test)

            # scores[i] = score

        print(f"Score: {...}")

    return ...  # scores.mean()
