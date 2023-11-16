import pickle

from data_processing import Data, Format_data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

""" Define training datasets """
# Load text datasets
print(f"Loading data files... ", end='')
filenames = ["MP2/data/train.csv", "MP2/data/test.csv"]
words_dataset = Data(train_file=filenames[0], test_file=filenames[1])
print(f'Done')

# Datasets
ds_options = [
    # Base dataset
    {
        'dataset_name': 'Base',
        'max_feat': 3000,
        'lemmatize': False,
        'lang_id': False,
        'feat_select': None,
    },
    {
        'dataset_name': '2G',
        'lang_id': False,
        'feature_type': 'Bin',
        'n_gram': (1, 2),
        'lemmatize': False,
        'feat_select': None,
    },
    {
        'dataset_name': 'Lem',
        'lang_id': False,
        'feature_type': 'Bin',
        'n_gram': (1, 2),
        'lemmatize': True,
        'feat_select': None,
    },
    {
        'dataset_name': 'No acc',
        'lang_id': False,
        'feature_type': 'Bin',
        'n_gram': (1, 2),
        'lemmatize': False,
        'feat_select': None,
        'rm_accents': True,
    },
    {
        'dataset_name': 'Spaces',
        'lang_id': False,
        'feature_type': 'Bin',
        'n_gram': (1, 2),
        'lemmatize': False,
        'feat_select': None,
        'rm_accents': False,
        'punc_replace': ' ',
    },
]

print(f"Processing input data...")
ds_list = []
for each_ds in ds_options:
    ds_list.append(Format_data(words_dataset, **each_ds))

print(f'Done')

print(f'Saving dataframe comp.')

df = words_dataset.train_data['body'].reset_index()
for ds in ds_list:
    df[ds.name] = [', '.join(x) for x in ds._vectorizer.inverse_transform(ds.X)]

df.drop(['index'], inplace=True, axis=1)
df.to_excel('MP2/preproc_analyse.xlsx', index=False)


# def my_preproc(text: str):
#     # words = [word for word in text.split() if '$' in word]
#     return text.lower()
#
#
# str = 'electricity 100$ consumed at night you could easily #ddd get a bill over £230'
#
# vectorizer = CountVectorizer(
#     stop_words=list(set(ds_list[0]._get_stop_words()) - {'$'}),
#     preprocessor=my_preproc,
#     # tokenizer=tokenizer,
#     # token_pattern=token_pattern,
#     # strip_accents=strip_accents,
# )
#
# vectorizer.fit_transform([str])
#
# print(vectorizer.get_feature_names_out())
#
# ...
