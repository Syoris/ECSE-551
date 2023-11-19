"""
To find the best model and their parameter combination using K-Fold validation
"""
import pickle

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from NaiveBayes import NaiveBayes, MyMultinomialNB

from cross_val_score import cross_val_score
from data_processing import Data, Format_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import itertools


# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


""" Define training datasets """
# Load text datasets
print(f"Loading data files... ", end='')
filenames = ["MP2/data/train.csv", "MP2/data/test.csv"]
words_dataset = Data(train_file=filenames[0], test_file=filenames[1])
print(f'Done')

# TODO: REMOVE
preds = pd.read_excel('MP2/pred_analysis.xlsx', sheet_name='test', usecols="A,C").set_index('id')  # noqa
preds.columns = ['True label']

ds_options = {
    'max_feat': [None],
    'lang_id': [False],  # [False, True],
    'feature_type': ['Bin'],  # Options: 'Bin', 'Count', 'TF'
    'rm_accents': [True],
    'n_gram': [(1, 1)],
    'lemmatize': [False],
    'feat_select': ['F_CL'],  # Options: 'PCA', 'MI', 'F_CL', None
    'n_feat_select': [1000],
    'weight_samples': [True],
}

# ds_options = {
#     'max_feat': [None],
#     'lang_id': [True, False],  # [False, True],
#     'feature_type': ['Count', 'Bin'],  # Options: 'Bin', 'Count', 'TF'
#     'rm_accents': [True],
#     'n_gram': [(1, 1), (1, 2), (1, 3), (1, 4)],
#     'lemmatize': [False],
#     'feat_select': ['F_CL'],  # Options: 'PCA', 'MI', 'F_CL', None
#     'n_feat_select': [1000, 2000, 3000],
#     'weight_samples': [False],
# }

print(f"Processing input data...")
keys, values = zip(*ds_options.items())
ds_options_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

ds_list = []
for idx, each_ds in enumerate(ds_options_list):
    each_ds['dataset_name'] = f'DS {idx}'
    ds_list.append(Format_data(words_dataset, **each_ds))

print(f'Done')


##
model_dict = {}
model_dict["My NB"] = {
    "model": NaiveBayes,
    'base_params': {'laplace_smoothing': True, 'verbose': False},
    'cv_params': None,
}

model_dict["MyMultinomialNB"] = {
    "model": MyMultinomialNB,
    'base_params': {},
    'cv_params': None,
}

model_dict["MultinomialNB"] = {
    "model": MultinomialNB,
    'base_params': {},
    'cv_params': None,
}
#
# model_dict["ComplementNB"] = {
#     "model": ComplementNB,
#     'base_params': {},
#     'cv_params': None,
# }
#
# model_dict["SVC"] = {
#     "model": LinearSVC,
#     "base_params": {"random_state": 0},
#     "cv_params": {"C": [0.001, 0.05, 0.1, 1]},
# }


def find_ds_from_name(ds_name, ds_list) -> Format_data:
    ds = next((ds for ds in ds_list if ds.name == ds_name), None)

    if ds is None:
        raise ValueError(f"Dataset {ds_name} not found in `ds_list`")

    return ds


def compute_models_cv_acc(model_dict, ds_list):
    # Load past results
    # try:
    #     with open('MP2/results.pkl', "rb") as file:
    #         results_df = pickle.load(file)
    # except FileNotFoundError:
    #     results_df = pd.DataFrame()
    results_df = pd.DataFrame()

    # Cross-Validation
    n_fold = 5

    start_time = time.time()
    print(f"--------- Training all models ---------")
    for model_name, model_info in model_dict.items():
        model = model_info["model"]
        base_params = model_info["base_params"]
        cv_params = model_info["cv_params"]

        print(f"\nModel : {model_name}")
        model_start = time.time()
        for ds_idx, each_dataset in enumerate(ds_list):
            try:
                # Check if it already has been ran
                ds_start = time.time()
                dataset_name = each_dataset.name
                print(f"\tDataset [{ds_idx+1}/{len(ds_list)}]: {dataset_name}")

                X_train = each_dataset.X
                y_train = each_dataset.Y

                # Cross_validation
                cv_results = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=n_fold,
                    base_params=base_params,
                    cv_params=cv_params,
                    results_df=results_df,
                    ds_name=dataset_name,
                    sample_weight=each_dataset.sample_weight,
                )

                if cv_results.empty:
                    print(f'... Model already trained')
                    continue

                # Print best combination
                best_row = cv_results.iloc[cv_results['Score'].idxmax()]
                compute_time = time.time() - ds_start
                print(
                    f"\tBest CV Score : {np.round(best_row['Score']*100)}% (Acc: {np.round(best_row['Acc']*100)}) "
                    f"[{compute_time} sec]\n"
                )

                # Add information to series
                ds_params = each_dataset.get_params()

                for key, value in ds_params.items():
                    if isinstance(value, tuple):
                        value = str(value)
                    cv_results[key] = value

                # cv_results = pd.concat([cv_results, pd.(ds_params).T], ignore_index=True)
                cv_results['Model name'] = model_name
                cv_results['Dataset'] = dataset_name
                cv_results['Compute time'] = compute_time

                results_df = pd.concat([results_df, cv_results], ignore_index=True, axis=0)

            except Exception as err:
                print(f"\n############## ERROR ##############")
                print(err)
                print(f"###################################")

        print(f'Model trained in {time.time() - model_start} sec')

    print(f"\nTraining completed ({time.time() - start_time} sec)\n")

    print(f'Saving results dataframe')
    results_df = results_df[
        [
            'Model name',
            'Score',
            'Acc',
            'Dataset',
            'Params',
            'Compute time',
            'Model',
            'n_gram',
            'feat_type',
            'lemmatized',
            'lang',
            'standardized',
            'rm_accents',
            'feat_select',
            'n_feat',
            'weight_samples',
        ]
    ]

    with open('MP2/results.pkl', "wb") as file:
        pickle.dump(results_df, file)

    return results_df


def create_pred_ds():
    with open('MP2/results.pkl', "rb") as file:
        results_df = pickle.load(file)

    results_df['Test Acc'] = 0.0

    for row_idx, each_model in results_df.iterrows():
        model = each_model['Model']
        ds = find_ds_from_name(each_model['Dataset'], ds_list)
        test_acc = model.score(ds.X_test, preds['True label'])

        results_df.at[row_idx, 'Test Acc'] = test_acc

    results_df.to_excel('MP2/results.xlsx')

    print(f"### Ordered Models ###")
    # print((results_df.sort_values(by=['Score'], ascending=False)).to_string())
    print((results_df.sort_values(by=['Test Acc', 'Score'], ascending=False)).to_string())

    model_idx = int(input(f"Input idx of model to use for test prediction: "))

    my_model_info = results_df.iloc[model_idx]
    print(f'Model chosen: ')
    print(my_model_info)

    print(f"Predicting test data using this model...")
    my_model = my_model_info['Model']
    ds = find_ds_from_name(my_model_info['Dataset'], ds_list)

    y_test = my_model.predict(ds.X_test)
    pred_df = pd.DataFrame(y_test, columns=['subreddit'])
    pred_df.index.name = 'id'
    pred_save_path = (
        f'MP2/predictions/my_pred_{int((my_model_info["Score"]*100).round())}'
        f'_{datetime.now().strftime(("%Y%m%d_%H%M"))}.csv'
    )
    pred_df.to_csv(pred_save_path)
    print(f'Predictions saved to {pred_save_path}')

    # ds.print_best_features(to_excel=True)


def check_nb_weights(model_info):
    model = model_info['Model']
    ds = find_ds_from_name(model_info['Dataset'], ds_list)
    if not isinstance(model, ComplementNB):
        print(f"Model isnt Naive Bayes")
        return

    feat_names = ds.features_name

    class_log_prior = model.class_log_prior_
    class_prior_prob = np.exp(class_log_prior)
    print(f"Prior prob of classes:")
    for c, c_prob in zip(model.classes_, class_prior_prob):
        print(f"\t-{c}: {c_prob}")

    feat_log_prob = model.feature_log_prob_
    feat_counts = model.feature_count_
    n_best_feat = 10

    for idx, c in enumerate(model.classes_):
        print(f"\n### Class: {c} ###")
        feats_score = feat_log_prob[idx, :]
        counts = feat_counts[idx, :]

        best_feats_idx = np.argsort(feats_score)[-n_best_feat:][::-1]
        most_counts_idx = np.argsort(counts)[-n_best_feat:][::-1]

        best_feats = feat_names[best_feats_idx]
        most_counts_features = feat_names[most_counts_idx]

        print(f"\tMost counts: {most_counts_features}")
        print(f"\tBest scores: {best_feats}")

        ...

    ...


if __name__ == '__main__':
    results_df = compute_models_cv_acc(model_dict, ds_list)
    create_pred_ds()


### TEMP UTILITIES ###
def get_test_features(ds_idx, post_idx):
    ds = ds_list[ds_idx]
    vect = ds._vectorizer
    selector = ds._feat_selector
    X_test = ds.X_test

    if selector is None:
        feats = vect.inverse_transform(X_test)
    else:
        # Get post words
        feats = vect.inverse_transform(selector.inverse_transform(X_test))

    return feats[post_idx]
