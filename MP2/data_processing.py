import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from nltk import word_tokenize
import pickle
from functools import partial

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif

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
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
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
        self.train_data = pd.read_csv(self.train_file, header=None, encoding='utf-8', skiprows=[0], names=['body', 'label'])


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

    def __init__(self, words_dataset: Data,
                 max_feat: int | None = 3000,
                 dataset_name:str = 'NoName',
                 n_gram: tuple = (1, 1),
                 binary_features: bool = True,  # If true, features are binary, false features are the frequency
                 use_tf_idf: bool = False,
                 feat_select: str | None = None, # Can be 'PCA', 'MI' or None
                 pca_n_components: int = 1,  # If >1, uses PCA
                 n_mi: int = 100,
                 lemmatize: bool = False,
                 lang_id:bool = False,  # If true, add a feature for the language (0: en, 1:fr)
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
        self.max_feat = max_feat
        self._n_gram = n_gram
        self._binary_features = binary_features
        self._use_tf_idf = use_tf_idf
        self._pca_n_components = pca_n_components
        self._lemmatize = lemmatize
        self._lang_id = lang_id
        self._feat_select_opt = feat_select
        self._n_mi = n_mi

        # Train labels
        self.Y = words_dataset.labels
        self.X_test = None

        # Pre-process
        self.train_text, self.test_text = self._pre_process_text()  # All samples in text format. (renamed from features)
        self.stop_words = self.get_stop_words()

        # Tokenize
        self.X, self.vectorizer = self.count_vectorizer()  # vectorizer: To transform text to a vector
        self.features_name = self.vectorizer.get_feature_names_out()  # Corresponding features of vectorizer

        self._add_lang()

        self.pca = None  # PCA transformer
        self.mi_selector = None  # MI feature selection
        self._feature_selection()


        self.transform_test()

        file_name = f'MP2/datasets/{self.name}.pkl'
        print(f" Saving to {file_name}")
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    def count_vectorizer(self):
        """
        Create a dictionary of all words.

        To get the CountVectorizer for the training dataset.

        Returns:
            vectorizer and vectorized dataset

        """
        # Set tokenizer
        if self._lemmatize:
            tokenizer = LemmaTokenizer()
            token_pattern = None
        else:
            tokenizer = None
            token_pattern = r'(?u)\b\w\w+\b'


        if not self._use_tf_idf:
            vectorizer = CountVectorizer(stop_words=self.stop_words,
                                         max_features=self.max_feat,
                                         ngram_range=self._n_gram,
                                         binary=self._binary_features,
                                         tokenizer=tokenizer,
                                         token_pattern=token_pattern
                                         )

        else:
            vectorizer = TfidfVectorizer(stop_words=self.stop_words,
                                         max_features=self.max_feat,
                                         ngram_range=self._n_gram,
                                         binary=False,
                                         tokenizer=tokenizer,
                                         token_pattern=token_pattern
                                         )

        # Learn the vocabulary dictionary and return document term matrix
        X = vectorizer.fit_transform(self.train_text)

        # print(vectorizer.get_feature_names_out())
        # X_bin = X.toarray()

        return X, vectorizer

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
    def get_stop_words(self):
        my_stop_words = stopwords.words('english') + stopwords.words('french')

        # Lemmatize stop words
        if self._lemmatize:
            wnl = WordNetLemmatizer()
            my_stop_words = [wnl.lemmatize(t, pos=get_wordnet_pos(t)) for t in my_stop_words if t.isalpha()]
            my_stop_words = list(set(my_stop_words))

        # print(my_stop_words)
        return my_stop_words

    # Remove punctuations from the dataset
    def _feature_selection(self):
        """
        To perform feature selection analysis
        """
        #PCA
        if self._feat_select_opt == 'PCA':
            if self._pca_n_components == 1:
                return

            self.pca = PCA(n_components=None)

            if isinstance(self.X, sp.csr_matrix):
                self.X = self.X.toarray()
            self.X = self.pca.fit_transform(self.X)

            plot = True
            if plot:
                sing_values = self.pca.singular_values_

                # Plot the singular values
                plt.plot(np.arange(1, len(sing_values) + 1), sing_values, marker='o')
                plt.title(f'Singular Values - {self.name}')
                plt.xlabel('Principal Components')
                plt.ylabel('Singular Values')
                plt.axvline(x=self._pca_n_components, color='red', linestyle='--', ymin=0, ymax=1, linewidth=2)
                plt.grid(True)
                plt.show(block=False)

        elif self._feat_select_opt == 'MI':
            if self._use_tf_idf:
                discrete_feat = [self.X.shape[1] - 1]
                X = self.X.toarray()
            else:
                discrete_feat = True
                X = self.X

            # MI_info = mutual_info_classif(X=self.X.toarray(), y=self.Y, discrete_features=discrete_features, random_state=0)
            my_score = partial(mutual_info_classif, random_state=0, discrete_features=discrete_feat)
            self.mi_selector = SelectKBest(my_score, k=self._n_mi)
            self.X = self.mi_selector.fit_transform(X, self.Y)

            self.features_name = self.mi_selector.get_feature_names_out(self.features_name)

        elif self._feat_select_opt is None:
            return

        else:
            raise ValueError(f'Invalid feature selction option: {self._feat_select_opt}')

    def _add_lang(self, transform_test=False):
        """
        Add a feature with the language of the post (en:0, fr:1)
        """
        if self._lang_id:
            if not transform_test:
                lang_array = (self.words_dataset.train_data['lang'] == 'fr').astype(int).to_numpy() # 0: en, 1:fr
                lang_sparse = sp.csr_matrix(lang_array).reshape(-1, 1)

                self.X = sp.csr_matrix(sp.hstack([self.X, lang_sparse]))

                self.features_name = np.append(self.features_name, 'lang')

            else:
                lang_array = (self.words_dataset.test_data['lang'] == 'fr').astype(int).to_numpy() # 0: en, 1:fr
                lang_sparse = sp.csr_matrix(lang_array).reshape(-1, 1)

                self.X_test = sp.csr_matrix(sp.hstack([self.X_test, lang_sparse]))

    def transform_test(self):
        # Tokenize
        self.X_test = self.vectorizer.transform(self.test_text)

        # Language
        self._add_lang(transform_test=True)

        # Feature selection
        if self._feat_select_opt == 'PCA':
            self.X_test = self.pca.transform(self.X_test.toarray())

        elif self._feat_select_opt == 'MI':
            self.X_test = self.mi_selector.transform(self.X_test)

        elif self._feat_select_opt is None:
            return

        return self.X_test
