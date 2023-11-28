import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import itertools
from NaiveBayes import MyMultinomialNB


def cross_val_score(
    model_class, X, y, cv=5, base_params=None, cv_params=None, results_df=None, ds_name=None, sample_weight=None
):
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
        pd.DataFrame: Score of each combination
    """
    kf = KFold(n_splits=cv, shuffle=True)

    results = []

    # Find all combinations of parameters
    if cv_params is not None:
        keys, values = zip(*cv_params.items())
        all_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        all_combs = [{}]

    # Find the best combination w/ CV
    for i, each_comb in enumerate(all_combs):
        print(f'\r\tCombination {i+1}/{len(all_combs)}', end='')
        model = model_class(**base_params, **each_comb)  # Create model with curr params combination
        # print(f"\tParams: {each_comb}", end='')

        # Check if model has already been trained on this ds
        if not results_df.empty:
            matching_row = results_df[
                (results_df['Model'].apply(type) == type(model))
                & (results_df['Params'] == each_comb)
                & (results_df['Dataset'] == ds_name)
            ]
            if not matching_row.empty:
                continue

        score = 0

        comb_ok = True  # Set to False if the combination of parameters is invalid

        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            if not comb_ok:
                break

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            try:
                if sample_weight is not None and isinstance(model, MyMultinomialNB):
                    score += model.fit(X_train, y_train, sample_weight=sample_weight).score(X_test, y_test)
                else:
                    score += model.fit(X_train, y_train).score(X_test, y_test)

            except ValueError as err:
                comb_ok = False
                err_msg = err

        score /= cv

        if comb_ok:
            # Train on whole ds
            if sample_weight is not None and isinstance(model, MyMultinomialNB):
                acc = model.fit(X, y, sample_weight=sample_weight).score(X, y)
            else:
                acc = model.fit(X, y).score(X, y)

            results.append({'Params': each_comb, 'Score': score, 'Model': model, 'Acc': acc})

        if not comb_ok:
            print(f"Invalid model: {err_msg}")

    return pd.DataFrame(results)
