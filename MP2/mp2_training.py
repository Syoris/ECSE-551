"""
To find the best model and their parameter combination using K-Fold validation
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from NaiveBayes import NaiveBayes
from cross_val_score import cross_val_score

from data_processing import Data, Format_data

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

#TODO
# - Viz important features
# - Predict test dataset

""" Define training datasets """
# Load text datasets
filenames = ["MP2/train.csv", "MP2/test.csv"]
words_to_train = Data(filenames[0], train = True)
words_to_test = Data(filenames[1], train = False)


max_features = 3000

dataset_list = []

print(f"Processing input data...", end='')
dataset_list.append(Format_data(words_to_train, max_feat=max_features, dataset_name='BASE'))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='TF', use_tf_idf=True))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='L', use_tf_idf=False, lemmatize=True))
# dataset_list.append(Format_data(words_to_train, max_feat=3000, dataset_name='L-TF', use_tf_idf=True, lemmatize=True))

# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='DS 1'))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='DS 2 - 2grams', n_gram=(1, 2)))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='DS 3 - 3grams', n_gram=(1, 3)))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='DS 4 - 2grams, tf', n_gram=(1, 2), use_tf_idf=True))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='DS 5 - 3grams, tf', n_gram=(1, 3), use_tf_idf=True))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='DS 6 - PCA 100', n_gram=(1, 2)))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='DS 7 - PCA 100', n_gram=(1, 2), use_tf_idf=True, pca_n_components=100))
# dataset_list.append(Format_data(words_to_train, max_feat=None, dataset_name='DS 8 - PCA 500', n_gram=(1, 2), use_tf_idf=True, pca_n_components=500))
print(f'\tDone')

##
model_dict = {}
# model_list["My NB"] = {
#     "model": NaiveBayes,
#     'base_params': {'laplace_smoothing': True, 'k_cv': 0, 'verbose': False},
#     'cv_params': None
# }

# model_list["SK Bernoulli NB"] = {
#     "model": BernoulliNB,
#     "train_data": train_dataset,
#     'base_params': {},
#     'cv_params': None
# }

# model_list["SVC"] = {
#     "model": svm.SVC,
#     "train_data": train_dataset,
#     "base_params": {"random_state": 0},
#     "cv_params": {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "C": [0.1, 1, 10]},
# }
#
model_dict['DT'] = {
    "model": tree.DecisionTreeClassifier,
    "base_params": {},
    # "cv_params": {"max_depth": [None, 100], "min_samples_split": [0.01, 0.005, 0.0001]},
    "cv_params": {"max_depth": [100], "min_samples_split": [0.0001]},
}

# model_list['Random Forest'] = {
#     "model": RandomForestClassifier,
#     "base_params": {},
#     "cv_params": None,
# }

# model_list['AdaBoost'] = {
#     "model": AdaBoostClassifier,
#     "base_params": {'n_estimators': 100, 'random_state': 0},
#     "cv_params": None,
# }

#TODO: Add KNN

# Cross-Validation
n_fold = 10
results_df = pd.DataFrame()

start_time = time.time()
print(f"--------- Training all models ---------")
for model_name, model_info in model_dict.items():
    model = model_info["model"]
    base_params = model_info["base_params"]
    cv_params = model_info["cv_params"]

    print(f"Model : {model_name}")
    model_start = time.time()
    for ds_idx, each_dataset in enumerate(dataset_list):
        ds_start = time.time()
        dataset_name = each_dataset.name
        print(f"\tDataset [{ds_idx}/{len(dataset_list)}]: {dataset_name}")

        X_train = each_dataset.X
        y_train = each_dataset.Y


        # Cross_validation
        cv_results = cross_val_score(
            model, X_train, y_train, cv=n_fold, base_params=base_params, cv_params=cv_params
        )

        best_row = cv_results.iloc[cv_results['Score'].idxmax()]
        compute_time = time.time() - ds_start
        print(f"\t\tBest Score : {best_row['Score']} with params: {best_row['Params']} [{compute_time} sec]\n")

        cv_results['Model'] = model_name
        cv_results['Dataset'] = dataset_name
        cv_results['Compute time'] = compute_time

        results_df = pd.concat([results_df, cv_results], ignore_index=True)

    print(f'Model trained in {time.time() - model_start} sec')

print(f"\nTraining completed ({time.time() - start_time} sec)\n")

print(f"\n\n--------- Processing the results ---------")
results_df = results_df[['Model', 'Score', 'Dataset', 'Compute time', 'Params']]

print(f"### Ordered Models ###")
print((results_df.sort_values(by=['Score'], ascending=False)).to_string())

# print(f'\n\n### Best of each model ###')
idx_max_scores = results_df.groupby('Model')['Score'].idxmax()
best_models_df = results_df.loc[idx_max_scores]
# print((best_models_df.sort_values(by=['Score'], ascending=False)).to_string())

# print(f"\n\n### Best Model ###")
best_model_data = best_models_df.loc[best_models_df['Score'].idxmax()]
# print(best_model)

print(f"### Training Best Model ###")
model_settings = model_dict[best_model_data['Model']]  # Get settings from model_dict
cv_params = best_model_data['Params']
best_ds = next((ds for ds in dataset_list if ds.name == best_model_data['Dataset']), None)
best_model = model_settings['model'](**model_settings['base_params'], **cv_params)

best_model.fit(best_ds.X, best_ds.Y)
final_score = best_model.score(best_ds.X, best_ds.Y)


print("\n------------------------")
print(f"BEST ACCURACY: {best_model_data['Score']*100}%")
print("------------------------")
