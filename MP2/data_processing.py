import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from nltk import word_tokenize
import pickle
from functools import partial
from typing import Literal
import unidecode

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn import preprocessing

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import langid
from langid.langid import LanguageIdentifier, model

langid.set_languages(['en', 'fr'])
lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t, pos=get_wordnet_pos(t)) for t in word_tokenize(doc) if t.isalpha()]


class Data:
    def __init__(self, train_file, test_file):
        # Download the csv data
        self.train_file: str = train_file
        self.test_file: str = test_file

        self.train_data: pd.DataFrame
        self.test_data: pd.DataFrame
        self.readData()
        self._detect_lang()

        # Extract the subsets
        self.data_list: list = self.train_data['body'].to_list()  # List of all samples

        self.labels = self.train_data['label'].to_numpy()  # Numpy array with label of each sample

    # Read the data
    def readData(self):
        self.train_data = pd.read_csv(
            self.train_file, header=None, encoding='utf-8', skiprows=[0], names=['body', 'label']
        )

        self.test_data = pd.read_csv(self.test_file, header=None, encoding='utf-8', skiprows=[0], names=['id', 'body'])

    def _detect_lang(self):
        """
        To find the language (fr or en) of each post
        """
        self.train_data['lang'] = self.train_data['body'].apply(lambda x: lang_identifier.classify(x)[0])
        self.test_data['lang'] = self.test_data['body'].apply(lambda x: lang_identifier.classify(x)[0])


class Format_data:
    # def __new__(cls, *args, **kwargs):
    #     """
    #     To load the dataset from a file, if it exists
    #     """
    #     file_path = f'MP2/datasets/{kwargs["dataset_name"]}.pkl'
    #     try:
    #         with open(file_path, 'rb') as f:
    #             print(f'Loading dataset from file')
    #             inst = pickle.load(f)
    #
    #         if not isinstance(inst, cls):
    #            raise TypeError('Unpickled object is not of type {}'.format(cls))
    #
    #     except FileNotFoundError:
    #         inst = super(Format_data, cls).__new__(cls, *args, **kwargs)
    #
    #     return inst

    def __init__(
        self,
        words_dataset: Data,  # Loaded data
        dataset_name: str = 'NoName',
        # Text processing options
        max_feat: int | None = 3000,  #  Max number of tokens
        n_gram: tuple = (1, 1),
        use_tf_idf: bool = False,
        binary_features: bool = True,  # If true, features are binary, false features are the frequency
        lemmatize: bool = False,
        lang_id: bool = False,  # If true, add a feature for the language (0: en, 1:fr)
        rm_accents: bool = False,  # To remove accents
        standardize_data: bool = True,  # To remove mean and std of all data
        # Feature selection options
        feat_select: Literal['PCA', 'MI'] | None = None,  # Can be 'PCA', 'MI' or None
        pca_n_feat: int = 1,  # If `feat_select` is PCA, number of features to keep
        mi_n_feat: int = 100,
    ):
        self.name: str = dataset_name
        print(f"\tProcessing of: {self.name}... ", end='')

        file_path = f'MP2/datasets/{self.name}.pkl'
        try:
            with open(file_path, 'rb') as f:
                print(f' Loading dataset from file')
                inst = pickle.load(f)

        except FileNotFoundError:
            inst = None

        if inst is not None:
            self.__dict__.update(inst.__dict__)
            return

        # Attributes
        self.words_dataset: Data = words_dataset

        # Text processing
        self._max_feat = max_feat
        self._n_gram = n_gram
        self._binary_features = binary_features
        self._use_tf_idf = use_tf_idf
        self._lemmatize = lemmatize
        self._lang_id = lang_id
        self._standardize_data = standardize_data
        self._rm_accents = rm_accents

        # Feature selection
        self._feat_select_opt = feat_select
        self._mi_n_feat = mi_n_feat
        self._pca_n_feat = pca_n_feat

        # Train labels
        self.Y = words_dataset.labels
        self.X_test = None

        self.stop_words = self._get_stop_words()

        # Pre-process
        (
            self.train_text,
            self.test_text,
        ) = self._pre_process_text()  # Get list of posts, lowered and w/o punctuations

        # Tokenize
        self.X, self.X_test, self._vectorizer = self._vectorize_text()  # _vectorizer: To transform text to a vector

        self.features_name = self._vectorizer.get_feature_names_out()  # Corresponding features of _vectorizer

        self._add_lang()  # Add language as a feature

        self._scaler = self._normalize_data()

        # Feat. Selection
        self.pca_selector = None  # PCA transformer
        self.mi_selector = None  # MI feature selection
        self._feat_selector = self._feature_selection()

        # Save dataset
        file_name = f'MP2/datasets/{self.name}.pkl'
        print(f" Saving to {file_name}")
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    def _vectorize_text(self):
        """
        Create a dictionary of all words.

        To get the CountVectorizer for the training dataset.

        Returns:
            _vectorizer and vectorized dataset

        """
        # Set tokenizer
        if self._lemmatize:
            tokenizer = LemmaTokenizer()
            token_pattern = None

        else:
            tokenizer = None
            token_pattern = r'(?u)\b\w\w+\b'  # Defined as default value, to remove warnings

        strip_accents = 'unicode' if self._rm_accents else None

        if not self._use_tf_idf:
            vectorizer = CountVectorizer(
                stop_words=self.stop_words,
                max_features=self._max_feat,
                ngram_range=self._n_gram,
                binary=self._binary_features,
                tokenizer=tokenizer,
                token_pattern=token_pattern,
                strip_accents=strip_accents,
            )

        else:
            vectorizer = TfidfVectorizer(
                stop_words=self.stop_words,
                max_features=self._max_feat,
                ngram_range=self._n_gram,
                binary=False,
                tokenizer=tokenizer,
                token_pattern=token_pattern,
                strip_accents=strip_accents,
            )

        # Learn the vocabulary dictionary and return document term matrix
        X = vectorizer.fit_transform(self.train_text)

        # Transform test data
        X_test = vectorizer.transform(self.test_text)

        return X, X_test, vectorizer

    def _pre_process_text(self):
        """
        Pre-process the texts:
            - Lowers everything
            - Remove punctuations

        Returns:
            [str]: List with all the post preprocessed

        """
        train_df = self.words_dataset.train_data
        test_df = self.words_dataset.test_data

        # Lower
        train_df['body'] = train_df['body'].str.lower()
        test_df['body'] = test_df['body'].str.lower()

        # Punctuation
        # punctuation_list = "?:.,;!"
        # punctuation_list = string.punctuation
        train_df['body'] = train_df['body'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
        test_df['body'] = test_df['body'].str.replace('[{}]'.format(string.punctuation), '', regex=True)

        return train_df['body'].to_list(), test_df['body'].to_list()

    # Specify stopwords
    def _get_stop_words(self):
        my_stop_words = stopwords.words('english') + stopwords.words('french')

        if self._rm_accents:
            my_stop_words = [unidecode.unidecode(word) for word in my_stop_words]

        # Lemmatize stop words
        if self._lemmatize:
            wnl = WordNetLemmatizer()
            my_stop_words = [wnl.lemmatize(t, pos=get_wordnet_pos(t)) for t in my_stop_words if t.isalpha()]
            my_stop_words = list(set(my_stop_words))

        # print(my_stop_words)
        return my_stop_words

    def _feature_selection(self):
        """
        To perform feature selection analysis
        """
        feat_selector = None
        # PCA
        if self._feat_select_opt == 'PCA':
            if self._pca_n_feat == 1:
                return

            pca_selector = PCA(n_components=None)

            if isinstance(self.X, sp.csr_matrix):
                self.X = self.X.toarray()
            self.X = pca_selector.fit_transform(self.X)
            self.X_test = pca_selector.transform(self.X_test)

            plot = True
            if plot:
                sing_values = pca_selector.singular_values_

                # Plot the singular values
                plt.plot(np.arange(1, len(sing_values) + 1), sing_values, marker='o')
                plt.title(f'Singular Values - {self.name}')
                plt.xlabel('Principal Components')
                plt.ylabel('Singular Values')
                plt.axvline(x=self._pca_n_feat, color='red', linestyle='--', ymin=0, ymax=1, linewidth=2)
                plt.grid(True)
                plt.show(block=False)

            feat_selector = pca_selector

        elif self._feat_select_opt == 'MI':
            if self._use_tf_idf:
                discrete_feat = [self.X.shape[1] - 1]
                X = self.X.toarray()
            else:
                discrete_feat = True
                X = self.X

            # MI_info = mutual_info_classif(X=self.X.toarray(), y=self.Y, discrete_features=discrete_features, random_state=0)
            my_score = partial(mutual_info_classif, random_state=0, discrete_features=discrete_feat)
            mi_selector = SelectKBest(my_score, k=self._mi_n_feat)
            self.X = mi_selector.fit_transform(X, self.Y)

            self.X_test = mi_selector.transform(self.X_test)

            self.features_name = mi_selector.get_feature_names_out(self.features_name)

            feat_selector = mi_selector

        elif self._feat_select_opt is None:
            return

        else:
            raise ValueError(f'Invalid feature selection option: {self._feat_select_opt}')

        return feat_selector

    def _add_lang(self):
        """
        Add a feature with the language of the post (en:0, fr:1)
        """
        if self._lang_id:
            # Train
            lang_array_train = (self.words_dataset.train_data['lang'] == 'fr').astype(int).to_numpy()  # 0: en,
            lang_sparse_train = sp.csr_matrix(lang_array_train).reshape(-1, 1)

            self.X = sp.csr_matrix(sp.hstack([self.X, lang_sparse_train]))
            self.features_name = np.append(self.features_name, 'lang')

            # Test
            lang_array_test = (self.words_dataset.test_data['lang'] == 'fr').astype(int).to_numpy()  # 0: en, 1:fr
            lang_sparse_test = sp.csr_matrix(lang_array_test).reshape(-1, 1)
            self.X_test = sp.csr_matrix(sp.hstack([self.X_test, lang_sparse_test]))

    def _normalize_data(self):
        """
        Remove mean and var of data
        """
        scaler = preprocessing.StandardScaler().fit(self.X.toarray())

        self.X = scaler.transform(self.X.toarray())
        self.X_test = scaler.transform(self.X_test.toarray())

        return scaler
