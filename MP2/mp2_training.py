"""
To find the best model and their parameter combination using K-Fold validation
"""
import pickle

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from NaiveBayes import NaiveBayes
from cross_val_score import cross_val_score
from data_processing import Data, Format_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime


# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# TODO
# - Viz important features
# - Predict test dataset

""" Define training datasets """
# Load text datasets
print(f"Loading data files... ", end='')
filenames = ["MP2/train.csv", "MP2/test.csv"]
words_dataset = Data(train_file=filenames[0], test_file=filenames[1])
print(f'Done')

max_features = 3000

# Datasets
ds_options = [
    # # Base dataset
    # {
    #     'dataset_name': 'Base',
    #     'max_feat': 3000,
    #     'lemmatize': False,
    #     'lang_id': False,
    #     'feat_select': None,
    # },
    # # Only with lang added
    # {
    #     'dataset_name': 'Lang',
    #     'max_feat': 3000,
    #     'lang_id': True,
    #     'lemmatize': False,
    #     'feat_select': None,
    #     'standardize_data': False,
    # },
    # # Lang + TF IDF normalized
    # {
    #     'dataset_name': 'TF IDF - Normalized',
    #     'max_feat': 3000,
    #     'lang_id': True,
    #     'lemmatize': False,
    #     'feat_select': None,
    #     'use_tf_idf': True,
    #     'standardize_data': True,
    # },
    # # 2G
    # {
    #     'dataset_name': '2G',
    #     'max_feat': 3000,
    #     'lang_id': True,
    #     'n_gram': (1, 2),
    #     'lemmatize': False,
    #     'feat_select': None,
    #     'use_tf_idf': False,
    #     'standardize_data': True,
    #     'rm_accents': False,
    # },
    # # 2G and TF-IDF
    # {
    #     'dataset_name': '2G TF',
    #     'max_feat': 3000,
    #     'lang_id': True,
    #     'n_gram': (1, 2),
    #     'lemmatize': False,
    #     'feat_select': None,
    #     'use_tf_idf': True,
    #     'standardize_data': True,
    #     'rm_accents': False,
    # },
    # Feature selection
    {
        'dataset_name': '3G - 1000 Ft',
        'max_feat': None,
        'lang_id': True,
        'n_gram': (1, 3),
        'lemmatize': False,
        'feat_select': 'F_CL',
        'n_feat_select': 1000,
        'use_tf_idf': True,
        'standardize_data': True,
        'rm_accents': False,
        'min_df': 2,
    },
    # Feature selection
    {
        'dataset_name': '3G - All Ft',
        'max_feat': None,
        'lang_id': True,
        'n_gram': (1, 3),
        'lemmatize': False,
        'feat_select': 'F_CL',
        'n_feat_select': 'all',
        'use_tf_idf': True,
        'standardize_data': True,
        'rm_accents': False,
        'min_df': 2,
    },
    # # TF IDF - No Max
    # {
    #     'dataset_name': 'TF IDF - No Max',
    #     'max_feat': 3000,
    #     'lang_id': True,
    #     'lemmatize': False,
    #     'feat_select': None,
    #     'use_tf_idf': True,
    #     'standardize_data': True,
    # },
    # MI 100
    # {
    #     'dataset_name': 'MI 100',
    #     'max_feat': max_features,
    #     'lemmatize': True,
    #     'lang_id': True,
    #     'feat_select': 'MI',
    #     'mi_n_feat': 100,
    # },
]

print(f"Processing input data...")
ds_list = []
for each_ds in ds_options:
    ds_list.append(Format_data(words_dataset, **each_ds))


# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='MI 500', lemmatize=True, lang_id=True, feat_select='MI', mi_n_feat=500))
# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='MI 1000', lemmatize=True, lang_id=True, feat_select='MI', mi_n_feat=1000))
#
#
# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='TF - MI 100', use_tf_idf=True, lemmatize=True, lang_id=True, feat_select='MI', mi_n_feat=100))
# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='TF - MI 500', use_tf_idf=True, lemmatize=True, lang_id=True, feat_select='MI', mi_n_feat=500))
# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='TF - MI 1000', use_tf_idf=True, lemmatize=True, lang_id=True, feat_select='MI', mi_n_feat=1000))
#
# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='TF - MI 500 - 2G', n_gram=(1, 2), use_tf_idf=True, lemmatize=True, lang_id=True, feat_select='MI', mi_n_feat=500))
# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='MI 500 - 2G', n_gram=(1, 2), use_tf_idf=False, lemmatize=True, lang_id=True, feat_select='MI', mi_n_feat=500))


# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='Le_L', lang_id=True, lemmatize=True))
#
# ds_list.append(Format_data(words_dataset, max_feat=max_features, dataset_name='Le_L_TF', lang_id=True, lemmatize=True, use_tf_idf=True))
#
# ds_list.append(Format_data(words_dataset, max_feat=None, dataset_name='Le_L_2g', n_gram=(1, 2), lemmatize=True, use_tf_idf=False, lang_id=True))

#
# ds_list.append(Format_data(words_dataset, max_feat=None, dataset_name='Le_L_TF_2g', n_gram=(1, 2), lemmatize=True, use_tf_idf=True, lang_id=True))
# ds_list.append(Format_data(words_dataset, max_feat=None, dataset_name='Le_L_TF_3g', n_gram=(1, 3), lemmatize=True, use_tf_idf=True, lang_id=True))
#
# ds_list.append(Format_data(words_dataset, max_feat=None, dataset_name='Le_L_TF_2g_PCA 100', n_gram=(1, 2), lemmatize=True, use_tf_idf=True, lang_id=True, pca_n_feat=100))
# ds_list.append(Format_data(words_dataset, max_feat=None, dataset_name='Le_L_TF_3g_PCA 100', n_gram=(1, 3), lemmatize=True, use_tf_idf=True, lang_id=True, pca_n_feat=100))
#
# ds_list.append(Format_data(words_dataset, max_feat=None, dataset_name='Le_L_TF_2g_PCA 500', n_gram=(1, 2), lemmatize=True, use_tf_idf=True, lang_id=True, pca_n_feat=500))
# ds_list.append(Format_data(words_dataset, max_feat=None, dataset_name='Le_L_TF_3g_PCA 500', n_gram=(1, 3), lemmatize=True, use_tf_idf=True, lang_id=True, pca_n_feat=500))


def print_features(ds: Format_data):
    if ds._feat_selector is None:
        print(ds.features_name)

    # See best features:


print(f'Done')


##
model_dict = {}
# model_dict["My NB"] = {
#     "model": NaiveBayes,
#     'base_params': {'laplace_smoothing': True, 'k_cv': 0, 'verbose': False},
#     'cv_params': None
# }

# model_dict["SK Bernoulli NB"] = {
#     "model": BernoulliNB,
#     "train_data": train_dataset,
#     'base_params': {},
#     'cv_params': None
# }


# model_dict["KNN"] = {
#     "model": KNeighborsClassifier,
#     "base_params": {},
#     "cv_params": {"n_neighbors": [3, 5, 9], "weights": ['distance']},
# }

# model_dict["SVC"] = {
#     "model": svm.SVC,
#     "base_params": {"random_state": 0},
#     "cv_params": {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "C": [0.1, 1, 10]},
# }

model_dict['DT'] = {
    "model": tree.DecisionTreeClassifier,
    "base_params": {'random_state': 0},
    # "cv_params": {"max_depth": [50, 100, 500], "min_samples_split": [0.0001]},
    "cv_params": {"max_depth": [50], "min_samples_split": [0.0001]},
}

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
#     "base_params": {'solver': 'adam', 'random_state': 0, 'max_iter': 1000},
#     # "cv_params": {"alpha": [1e-5], 'hidden_layer_sizes': [(5, 2), (10, 5, 2), (50, 10), (3, 2)]},
#     "cv_params": {"alpha": [1e-5], 'hidden_layer_sizes': [(10, 5, 2)]},
# }


def find_best_model():
    # Cross-Validation
    n_fold = 10
    results_df = pd.DataFrame()

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
                ds_start = time.time()
                dataset_name = each_dataset.name
                print(f"\tDataset [{ds_idx+1}/{len(ds_list)}]: {dataset_name}")

                X_train = each_dataset.X
                y_train = each_dataset.Y

                # Cross_validation
                cv_results = cross_val_score(
                    model, X_train, y_train, cv=n_fold, base_params=base_params, cv_params=cv_params
                )

                best_row = cv_results.iloc[cv_results['Score'].idxmax()]
                compute_time = time.time() - ds_start
                print(f"\tBest Score : {best_row['Score']} [{compute_time} sec]\n")

                cv_results['Model name'] = model_name
                cv_results['Dataset'] = dataset_name
                cv_results['Compute time'] = compute_time

                results_df = pd.concat([results_df, cv_results], ignore_index=True)

            except Exception as err:
                print(f"############## ERROR ##############")
                print(err)
                print(f"###################################")

        print(f'Model trained in {time.time() - model_start} sec')

    print(f"\nTraining completed ({time.time() - start_time} sec)\n")

    results_df = results_df[['Model name', 'Score', 'Acc', 'Dataset', 'Params', 'Compute time', 'Model']]
    with open('MP2/results.pkl', "wb") as file:
        pickle.dump(results_df, file)


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

    print(f"### Training Best Model ###")
    best_model = best_model_data['Model']
    best_ds = next((ds for ds in ds_list if ds.name == best_model_data['Dataset']), None)

    best_model_score = best_model_data['Score'] * 100
    best_model_acc = best_model_data['Acc'] * 100

    print("\n------------------------")
    print(f"CV SCORE: {best_model_score}%")
    print(f"ACCURACY: {best_model_acc}%")
    print("------------------------")

    print(f"### Test Data Prediction ###")
    y_test = best_model.predict(best_ds.X_test)
    pred_df = pd.DataFrame(y_test, columns=['subreddit'])
    pred_df.index.name = 'id'
    pred_save_path = (
        f'MP2/predictions/pred_{int(best_model_score.round())}_{datetime.now().strftime(("%y%m%d_%H%M"))}.csv'
    )
    pred_df.to_csv(pred_save_path)
    print(f'Predictions saved to {pred_save_path}')


if __name__ == '__main__':
    find_best_model()
    process_results_and_predict()
