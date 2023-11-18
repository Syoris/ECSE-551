"""
To find the best model and their parameter combination using K-Fold validation
"""
import pickle

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn import svm
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

max_features = 3000

# NAMING
# With lang (L, nL) - Feature type (B, C, TF) - Lem (Le, NLe) - N-Gram (1G, 2G, 3G) - Feat select (FX, PX, MX)

# Datasets
# ds_options = [
#     # # Base dataset
#     # {
#     #     'dataset_name': 'Base',
#     #     'max_feat': 3000,
#     #     'lemmatize': False,
#     #     'lang_id': False,
#     #     'feat_select': None,
#     # },
#     # # # Only with lang added
#     # # {
#     # #     'dataset_name': 'L - B - NLe - 1G',
#     # #     'lang_id': True,
#     # #     'feature_type': 'Bin',
#     # #     'lemmatize': False,
#     # #     'feat_select': None,
#     # #     'standardize_data': False,
#     # # },
#     # # # Lang + TF IDF
#     # # {
#     # #     'dataset_name': 'L - TF - NLe - 1G',
#     # #     'lang_id': True,
#     # #     'feature_type': 'TF',
#     # #     'lemmatize': False,
#     # #     'feat_select': None,
#     # # },
#     # # # Count
#     # # {
#     # #     'dataset_name': 'L - C - NLe - 1G',
#     # #     'max_feat': 3000,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': False,
#     # #     'feat_select': None,
#     # # },
#     # # # 2G
#     # {
#     #     'dataset_name': 'NL - B - NLe - 2G',
#     #     'lang_id': False,
#     #     'feature_type': 'Bin',
#     #     'n_gram': (1, 2),
#     #     'lemmatize': False,
#     #     'feat_select': None,
#     # },
#     # {
#     #     'dataset_name': 'NL - C - NLe - 2G',
#     #     'lang_id': False,
#     #     'feature_type': 'Count',
#     #     'n_gram': (1, 2),
#     #     'lemmatize': False,
#     #     'feat_select': None,
#     # },
#     # # # 2G and TF-IDF
#     # # {
#     # #     'dataset_name': 'L - TF - NLe - 2G',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'TF',
#     # #     'n_gram': (1, 2),
#     # #     'lemmatize': False,
#     # #     'feat_select': None,
#     # # },
#     # # # 2G - LEM
#     # # {
#     # #     'dataset_name': 'L - C - Le - 2G',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'n_gram': (1, 2),
#     # #     'lemmatize': True,
#     # #     'feat_select': None,
#     # # },
#     # # # 2G, LEM, TF-IDF
#     # # {
#     # #     'dataset_name': 'L - TF - Le - 2G',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'TF',
#     # #     'n_gram': (1, 2),
#     # #     'lemmatize': True,
#     # #     'feat_select': None,
#     # # },
#     # # # 2G, LEM, TF-IDF
#     # # {
#     # #     'dataset_name': 'L - TF - Le - 2G - F100',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'TF',
#     # #     'n_gram': (1, 2),
#     # #     'lemmatize': True,
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 100,
#     # # },
#     # # # 2G, With Feature selections
#     # # {
#     # #     'dataset_name': 'L - B - NLe - 2G - F100',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Bin',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 100,
#     # # },
#     # # {
#     # #     'dataset_name': 'L - C - NLe - 2G - F250',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 250,
#     # # },
#     # # {
#     # #     'dataset_name': 'L - C - NLe - 2G - F500',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 500,
#     # # },
#     # # {
#     # #     'dataset_name': 'L - C - NLe - 2G - F1000',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 1000,
#     # # },
#     # # {
#     # #     'dataset_name': 'L - B - NLe - 2G',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Bin',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': None,
#     # # },
#     # {
#     #     'dataset_name': 'Big test',
#     #     'max_feat': None,
#     #     'lang_id': True,
#     #     'feature_type': 'Count',
#     #     'lemmatize': False,
#     #     'n_gram': (1, 2),
#     #     'feat_select': 'F_CL',
#     #     'n_feat_select': 2000,
#     #     'rm_accents': True,
#     # },
#     # {
#     #     'dataset_name': 'Big test- BIN',
#     #     'max_feat': None,
#     #     'lang_id': True,
#     #     'feature_type': 'Bin',
#     #     'lemmatize': False,
#     #     'n_gram': (1, 2),
#     #     'feat_select': 'F_CL',
#     #     'n_feat_select': 2000,
#     #     'rm_accents': True,
#     # },
#     # {
#     #     'dataset_name': 'Big test- BIN - NL',
#     #     'max_feat': None,
#     #     'lang_id': False,
#     #     'feature_type': 'Bin',
#     #     'lemmatize': False,
#     #     'n_gram': (1, 2),
#     #     'feat_select': 'F_CL',
#     #     'n_feat_select': 2000,
#     #     'rm_accents': True,
#     # },
#     # {
#     #     'dataset_name': 'Big test- Count - NL',
#     #     'max_feat': None,
#     #     'lang_id': False,
#     #     'feature_type': 'Count',
#     #     'lemmatize': False,
#     #     'n_gram': (1, 2),
#     #     'feat_select': 'F_CL',
#     #     'n_feat_select': 2000,
#     #     'rm_accents': True,
#     # },
#     # # {
#     # #     'dataset_name': 'L - B - NLe - 2G - F2000',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Bin',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 2000,
#     # # },
#     # # {
#     # #     'dataset_name': 'L - TF - NLe - 2G - F2000',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'TF',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 2000,
#     # # },
#     # # {
#     # #     'dataset_name': 'L - C - NLe - 2G - F2000 - No Acc',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 2000,
#     # #     'rm_accents': True,
#     # # },
#     # # {
#     # #     'dataset_name': 'L - C - NLe - 2G - F2000 - Punc w space',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 2000,
#     # #     'rm_accents': False,
#     # #     'punc_replace': ' ',
#     # # },
#     # # {
#     # #     'dataset_name': 'L - C - Le - 2G - F2000',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': True,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 2000,
#     # #     'rm_accents': False,
#     # #     'punc_replace': '',
#     # # },
#     # # {
#     # #     'dataset_name': 'L - C - NLe - 2G - F3000',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 3000,
#     # # },
#     # # {
#     # #     'dataset_name': 'L - C - Le - 2G - F2000',
#     # #     'max_feat': None,
#     # #     'lang_id': True,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': True,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 2000,
#     # # },
#     # # Feature selection - No Lang
#     # # {
#     # #     'dataset_name': 'NL - C - NLe - 2G - F2000',
#     # #     'max_feat': None,
#     # #     'lang_id': False,
#     # #     'feature_type': 'Count',
#     # #     'lemmatize': False,
#     # #     'n_gram': (1, 2),
#     # #     'feat_select': 'F_CL',
#     # #     'n_feat_select': 2000,
#     # # },
# ]

# ds_options = {
#     'max_feat': [None],
#     'lang_id': [True],  # [False, True],
#     'feature_type': ['Count', 'Bin'],  # Options: 'Bin', 'Count', 'TF'
#     'rm_accents': [True],
#     'n_gram': [(1, 1), (1, 2)],
#     'lemmatize': [False],
#     'feat_select': ['F_CL'],  # Options: 'PCA', 'MI', 'F_CL'
#     'n_feat_select': [100, 500, 1000, 2000],
#     'weight_samples': [True],
# }

ds_options = {
    'max_feat': [None],
    'lang_id': [False],  # [False, True],
    'feature_type': ['Bin'],  # Options: 'Bin', 'Count', 'TF'
    'rm_accents': [True],
    'n_gram': [(1, 2)],
    'lemmatize': [False],
    'feat_select': ['F_CL'],  # Options: 'PCA', 'MI', 'F_CL', None
    'n_feat_select': [2000],
    'weight_samples': [False],
}

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
# model_dict["My NB"] = {
#     "model": NaiveBayes,
#     'base_params': {'laplace_smoothing': True, 'k_cv': 0, 'verbose': False},
#     'cv_params': None
# }
#
# model_dict["SK Bernoulli NB"] = {
#     "model": BernoulliNB,
#     "train_data": train_dataset,
#     'base_params': {},
#     'cv_params': None
# }
#
# model_dict["MultinomialNB"] = {
#     "model": MultinomialNB,
#     'base_params': {},
#     'cv_params': None,
# }
#
# model_dict["ComplementNB"] = {
#     "model": ComplementNB,
#     'base_params': {},
#     'cv_params': None,
# }

# model_dict["ComplementNB"] = {
#     "model": ComplementNB,
#     'base_params': {},
#     'cv_params': None,
# }

model_dict["MultinomialNB"] = {
    "model": MultinomialNB,
    'base_params': {},
    'cv_params': None,
}

model_dict["MyMultinomialNB"] = {
    "model": MyMultinomialNB,
    'base_params': {},
    'cv_params': None,
}


# model_dict["KNN"] = {
#     "model": KNeighborsClassifier,
#     "base_params": {},
#     # "cv_params": {"n_neighbors": [3, 5, 9], "weights": ['distance']},
#     "cv_params": {"n_neighbors": [3], "weights": ['distance']},
# }

# model_dict["SVC"] = {
#     "model": svm.SVC,
#     "base_params": {"random_state": 0},
#     "cv_params": {"kernel": ['linear'], "C": [0.1, 1, 10]},
# }
#
# model_dict['DT'] = {
#     "model": tree.DecisionTreeClassifier,
#     "base_params": {'random_state': 0},
#     "cv_params": {
#         "max_depth": [5, 50, 100, 250],
#         "min_samples_split": [3, 5],
#     },
#     # "cv_params": {"max_depth": [50], "min_samples_split": [0.0001]},
# }

# model_dict['Random Forest'] = {
#     "model": RandomForestClassifier,
#     "base_params": {},
#     "cv_params": None,
# }
#
# model_dict['AdaBoost'] = {
#     "model": AdaBoostClassifier,
#     "base_params": {'random_state': 0},
#     "cv_params": {"n_estimators": [50, 100, 500]},
# }

# model_dict['MLP'] = {
#     "model": MLPClassifier,
#     "base_params": {'solver': 'adam', 'random_state': 0, 'max_iter': 2000},
#     "cv_params": {
#         "alpha": [0.01],
#         'hidden_layer_sizes': [(512, 256), (256, 128), (128, 64), (64, 32), (32, 16), (16, 8), (8, 4), (4, 2), (2, 1)],
#     },
# }


def find_ds_from_name(ds_name) -> Format_data:
    ds = next((ds for ds in ds_list if ds.name == ds_name), None)

    if ds is None:
        raise ValueError(f"Dataset {ds_name} not found in `ds_list`")

    return ds


def find_best_model():
    # Load past results
    # try:
    #     with open('MP2/results.pkl', "rb") as file:
    #         results_df = pickle.load(file)
    # except FileNotFoundError:
    #     results_df = pd.DataFrame()
    results_df = pd.DataFrame()

    # Cross-Validation
    n_fold = 5
    n_loops = 1

    # results_df = pd.DataFrame()

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

                cv_results = pd.DataFrame()
                results_list = []
                for i in range(n_loops):
                    # Cross_validation
                    cv_results_i = cross_val_score(
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

                    if cv_results_i.empty:
                        break

                    results_list.append(cv_results_i)

                if not results_list:
                    print(f'... Model already trained')
                    continue

                # Combine all iterations
                cv_results = pd.concat(results_list)
                cv_results = cv_results.groupby(['Acc'], as_index=False).agg(
                    {'Score': 'mean', 'Model': 'first', 'Params': 'first'}
                )

                # Print best combination
                best_row = cv_results.iloc[cv_results['Score'].idxmax()]
                compute_time = time.time() - ds_start
                print(
                    f"\tBest CV Score : {np.round(best_row['Score']*100)}% (Acc: {np.round(best_row['Acc']*100)}) "
                    f"[{compute_time} sec]\n"
                )

                # Add information to series
                ds_params = each_dataset.get_params()
                cv_results = pd.concat([cv_results.iloc[0], pd.Series(ds_params)])
                cv_results['Model name'] = model_name
                cv_results['Dataset'] = dataset_name
                cv_results['Compute time'] = compute_time

                results_df = pd.concat([results_df, cv_results.to_frame().T], ignore_index=True, axis=0)

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
        ]
    ]

    with open('MP2/results.pkl', "wb") as file:
        pickle.dump(results_df, file)
    #
    # results_df.to_excel('MP2/results.xlsx')


def process_results_and_predict():
    with open('MP2/results.pkl', "rb") as file:
        results_df = pickle.load(file)

    print(f"\n\n--------- Processing the results ---------")
    print(f"### Ordered Models ###")
    print((results_df.sort_values(by=['Score'], ascending=False)).to_string())

    # print(f'\n\n### Best of each model ###')
    idx_max_scores = results_df.groupby('Model name')['Score'].idxmax()
    best_models_df = results_df.loc[idx_max_scores]  # Best of each model
    # print((best_model_df.sort_values(by=['Score'], ascending=False)).to_string())

    # print(f"\n\n### Best Model ###")
    best_model_data = best_models_df.loc[best_models_df['Score'].idxmax()]
    # print(best_model)

    # print(f"### Training Best Model ###")
    # best_model = best_model_data['Model']
    # best_ds = next((ds for ds in ds_list if ds.name == best_model_data['Dataset']), None)
    #
    # best_model_score = best_model_data['Score'] * 100
    # best_model_acc = best_model_data['Acc'] * 100
    #
    # print("\n------------------------")
    # print(f"CV SCORE: {best_model_score}%")
    # print(f"ACCURACY: {best_model_acc}%")
    # print("------------------------")

    # print(f"### Test Data Prediction ###")
    # y_test = best_model.predict(best_ds.X_test)
    # pred_df = pd.DataFrame(y_test, columns=['subreddit'])
    # pred_df.index.name = 'id'
    # pred_save_path = (
    #     f'MP2/predictions/pred_{int(best_model_score.round())}_{datetime.now().strftime(("%Y%m%d_%H%M"))}.csv'
    # )
    # pred_df.to_csv(pred_save_path)
    # print(f'Predictions saved to {pred_save_path}')


def create_pred_ds():
    with open('MP2/results.pkl', "rb") as file:
        results_df = pickle.load(file)

    print(f"### Ordered Models ###")
    print((results_df.sort_values(by=['Score'], ascending=False)).to_string())

    model_idx = int(input(f"Input idx of model to use for test prediction: "))

    my_model_info = results_df.iloc[model_idx]
    print(f'Model chosen: ')
    print(my_model_info)

    print(f"Predicting test data using this model...")
    my_model = my_model_info['Model']
    ds = find_ds_from_name(my_model_info['Dataset'])

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


def check_results():
    with open('MP2/results.pkl', "rb") as file:
        results_df = pickle.load(file)
    results_df = results_df.sort_values(by=['Score'], ascending=False)
    bias_tol = 0.0  # In %

    df = results_df[results_df['Acc'] < (1.0 - bias_tol / 100)]
    df = df.sort_values(by=['Score'], ascending=False)

    print(df.to_string())
    ...


def check_nb_weights(model_info):
    model = model_info['Model']
    ds = find_ds_from_name(model_info['Dataset'])
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
    find_best_model()
    # check_results()
    create_pred_ds()

    # process_results_and_predict()
    ...

    # with open('MP2/results.pkl', "rb") as file:
    #     results_df = pickle.load(file)
    # results_df = results_df.sort_values(by=['Score'], ascending=False)
    # check_nb_weights(results_df.iloc[1])


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
