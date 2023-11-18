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

MY_STOP_WORDS = ['im', 'https', 'http', 'www', 'l', 're', 'qu', 'x200b']


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


def MyTokenizer(text):
    """
    To keep $ and pound signs
    """
    text = text.split()

    important_symbols = ['$', '£', '€']
    for symb in important_symbols:
        if any(symb in string for string in text):
            text.append(symb)

    return text


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
        self.train_data = self.train_data.sample(frac=1, random_state=0)
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
        max_feat: int | None = None,  #  Max number of tokens
        feature_type: Literal['Bin', 'Count', 'TF'] = 'Bin',
        n_gram: tuple = (1, 1),
        lemmatize: bool = False,
        lang_id: bool = False,  # If true, add a feature for the language (0: en, 1:fr)
        rm_accents: bool = False,  # To remove accents
        standardize_data: bool = False,  # To remove mean and std of all data
        min_df: int = 1,  # Ignore terms w/ frequency lower than that
        # Feature selection options
        feat_select: Literal['PCA', 'MI', 'F_CL'] | None = None,
        n_feat_select: int = 1,  # Number of features to keep
        weight_samples: bool = False,  # To compute the features weights
        punc_replace: str = ' ',
    ):
        self.name: str = dataset_name
        print(f"\tProcessing of: {self.name}... ", end='')

        # Attributes
        self.words_dataset: Data = words_dataset

        # Text processing
        self._max_feat = max_feat
        self._n_gram = n_gram

        self._feature_type = feature_type
        if feature_type == 'Bin':
            self._binary_features = True
            self._use_tf_idf = False

        elif feature_type == 'Count':
            self._binary_features = False
            self._use_tf_idf = False

        elif feature_type == 'TF':
            self._binary_features = False
            self._use_tf_idf = True

        self._lemmatize = lemmatize
        self._lang_id = lang_id
        self._standardize_data = standardize_data
        self._rm_accents = rm_accents
        self._min_df = min_df
        self._punc_rep = punc_replace
        self._weight_samples = weight_samples

        # Feature selection
        self._feat_select_opt = feat_select
        self._n_feat_select = n_feat_select

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

        self.sample_weight = None
        if self._weight_samples:
            self.sample_weight = self._compute_sample_weights()

        # self.print_best_features()

        print(f' Done')

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

        else:
            tokenizer = MyTokenizer

        strip_accents = 'unicode' if self._rm_accents else None

        if not self._use_tf_idf:
            vectorizer = CountVectorizer(
                stop_words=self.stop_words,
                max_features=self._max_feat,
                ngram_range=self._n_gram,
                binary=self._binary_features,
                tokenizer=tokenizer,
                token_pattern=None,
                strip_accents=strip_accents,
                min_df=self._min_df,
            )

        else:
            vectorizer = TfidfVectorizer(
                stop_words=self.stop_words,
                max_features=self._max_feat,
                ngram_range=self._n_gram,
                binary=False,
                tokenizer=tokenizer,
                token_pattern=None,
                strip_accents=strip_accents,
                min_df=self._min_df,
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
        train_df = self.words_dataset.train_data.copy(deep=True)
        test_df = self.words_dataset.test_data.copy(deep=True)

        # Lower
        train_df['body'] = train_df['body'].str.lower()
        test_df['body'] = test_df['body'].str.lower()

        # Punctuation
        # punctuation_list = "?:.,;!"
        # punctuation_list = string.punctuation
        punc_list = string.punctuation.replace('$', '')
        punc_list += '’«»“”'
        # Remove $ from the punctuations
        train_df['body'] = train_df['body'].str.replace('[{}]'.format(punc_list), self._punc_rep, regex=True)
        train_df['body'] = train_df['body'].str.replace(r'[\n\\]', '', regex=True)

        test_df['body'] = test_df['body'].str.replace('[{}]'.format(punc_list), self._punc_rep, regex=True)
        test_df['body'] = test_df['body'].str.replace(r'[\n\\]', '', regex=True)

        return train_df['body'].to_list(), test_df['body'].to_list()

    # Specify stopwords
    def _get_stop_words(self):
        my_stop_words = stopwords.words('english') + stopwords.words('french')

        my_stop_words += MY_STOP_WORDS

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
            pca_selector = PCA(n_components=self._n_feat_select)

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
                plt.axvline(x=self._n_feat_select, color='red', linestyle='--', ymin=0, ymax=1, linewidth=2)
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

            # MI_info = mutual_info_classif(X=self.X.toarray(), Y=self.Y, discrete_features=discrete_features, random_state=0)
            my_score = partial(mutual_info_classif, random_state=0, discrete_features=discrete_feat)
            mi_selector = SelectKBest(my_score, k=self._n_feat_select)
            self.X = mi_selector.fit_transform(X, self.Y)
            self.X_test = mi_selector.transform(self.X_test)

            selected_feats = self.features_name[mi_selector.get_support()]
            feat_scores = mi_selector.scores_[mi_selector.get_support()]
            names_scores = list(zip(selected_feats, feat_scores))
            feat_scores = pd.DataFrame(data=names_scores, columns=['Feat_names', 'Score'])
            self._feat_scores = feat_scores.sort_values(['Score', 'Feat_names'], ascending=[False, True])

            self.features_name = mi_selector.get_feature_names_out(self.features_name)

            feat_selector = mi_selector

        elif self._feat_select_opt == 'F_CL':
            # if self._use_tf_idf:
            #     discrete_feat = [self.X.shape[1] - 1]
            #     X = self.X.toarray()
            #
            # else:
            #     discrete_feat = True
            #     X = self.X

            # MI_info = mutual_info_classif(X=self.X.toarray(), Y=self.Y, discrete_features=discrete_features, random_state=0)
            feat_selector = SelectKBest(f_classif, k=self._n_feat_select)
            X_trans = feat_selector.fit_transform(self.X, self.Y)

            X_test_trans = feat_selector.transform(self.X_test)

            selected_feats = self.features_name[feat_selector.get_support()]
            feat_scores = feat_selector.scores_[feat_selector.get_support()]
            names_scores = list(zip(selected_feats, feat_scores))
            feat_scores = pd.DataFrame(data=names_scores, columns=['Feat_names', 'Score'])
            self._feat_scores = feat_scores.sort_values(['Score', 'Feat_names'], ascending=[False, True])

            self.features_name = feat_selector.get_feature_names_out(self.features_name)

            self.X, self.X_test = X_trans, X_test_trans

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
            en_train_array = (self.words_dataset.train_data['lang'] == 'en').astype(int).to_numpy()  # 0: en,
            en_train_array = sp.csr_matrix(en_train_array).reshape(-1, 1)

            fr_train_array = (self.words_dataset.train_data['lang'] == 'fr').astype(int).to_numpy()  # 0: en,
            fr_train_array = sp.csr_matrix(fr_train_array).reshape(-1, 1)

            self.X = sp.csr_matrix(sp.hstack([self.X, en_train_array, fr_train_array]))
            self.features_name = np.append(self.features_name, ['is_en', 'is_fr'])

            # Test
            en_test_array = (self.words_dataset.test_data['lang'] == 'en').astype(int).to_numpy()  # 0: en,
            en_test_array = sp.csr_matrix(en_test_array).reshape(-1, 1)

            fr_test_array = (self.words_dataset.test_data['lang'] == 'fr').astype(int).to_numpy()  # 0: en,
            fr_test_array = sp.csr_matrix(fr_test_array).reshape(-1, 1)

            self.X_test = sp.csr_matrix(sp.hstack([self.X_test, en_test_array, fr_test_array]))

    def _normalize_data(self):
        """
        Remove mean and var of data
        """

        scaler = None
        if self._standardize_data:
            scaler = preprocessing.StandardScaler().fit(self.X.toarray())

            self.X = scaler.transform(self.X.toarray())
            self.X_test = scaler.transform(self.X_test.toarray())

        return scaler

    def get_params(self):
        return {
            # 'max_feat': self._max_feat,
            'n_gram': self._n_gram,
            'feat_type': self._feature_type,
            'lemmatized': self._lemmatize,
            'lang': self._lang_id,
            'standardized': self._standardize_data,
            'rm_accents': self._rm_accents,
            'feat_select': self._feat_select_opt,
            'n_feat': self._n_feat_select,
        }

    def _compute_sample_weights(self):
        X = self.X
        Y = self.Y

        # # Analyze classes
        # classes, class_count = np.unique(Y, return_counts=True)
        # n_class = len(classes)
        # n_sample, n_features = X.shape
        #
        # # Merge documents
        # merged_counts = np.zeros([n_class, n_features])
        # # merged_counts = np.array([[1, 0, 0], [20, 1, 0], [20, 20, 1]])  # Rows: classes, cols: features
        #
        # # for each class k
        # for k, class_label in enumerate(classes):
        #     c_count = class_count[k]
        #
        #     X_k = X[Y == class_label, :]  # Select rows of class k
        #
        #     merged_counts[k, :] = X_k.sum(axis=0)
        #
        # # Compute sample weights
        # total_counts = int(merged_counts.sum())
        # weights = np.zeros([n_class, n_features])
        # for k, class_label in enumerate(classes):
        #     n_c = n_class  # Number of classes
        #     n_a_c = merged_counts[k, :].sum()
        #
        #     # for each feature j
        #     for j in range(n_features):
        #         n_aj = merged_counts[:, j].sum()  # Total number of count of feature j
        #         p_aj = n_aj / total_counts  # P(a_j)
        #
        #         n_aj_c = merged_counts[k, j]  # Number of count of feature j in class k
        #         p_aj_kw_c = n_aj_c / n_a_c  # Prob(a_j | c)
        #
        #         R_aj_c = p_aj_kw_c / p_aj
        #
        #         p_aj_c = n_aj_c / total_counts  # Prob documents in class and contain ai
        #
        #         # Compute CR_ai_c : Weight of feature a_i for class c
        #
        #         k_ai = (merged_counts[:, j] != 0).sum()  # Number of classes of documents that contain ai
        #         # Greater k_ai, smaller is the dep. bw ai and class c
        #
        #         # p_ai_c / p_ai  Class dist. of the documents with ai. Greater it is, greater the dep. bw ai and class c
        #
        #         CR_aj_c = R_aj_c * p_aj_c / p_aj * np.log(2 + n_c / k_ai)
        #         weights[k, j] = CR_aj_c
        #
        # # Multiply features by weights
        # # for each class k
        # for k, class_label in enumerate(classes):
        #     if not isinstance(self.X, np.ndarray):
        #         self.X = self.X.toarray()
        #
        #     # if not isinstance(self.X_test, np.ndarray):
        #     #     self.X_test = self.X_test.toarray()
        #
        #     self.X[Y == class_label, :] = self.X[Y == class_label, :] * weights[k, :]
        #     # self.X_test[Y == class_label, :] = self.X_test[Y == class_label, :] * weights[k, :]

        if not isinstance(self.X, np.ndarray):
            self.X = self.X.toarray()

        if not isinstance(self.X_test, np.ndarray):
            self.X_test = self.X_test.toarray()

        weights = self._feat_scores.sort_index()['Score'].to_numpy().copy()
        # weights /= weights.sum()

        self.X = self.X * weights.reshape(1, -1)
        self.X_test = self.X_test * weights

        return weights

    def print_best_features(self, n_feats=10, to_excel=False):
        """
        To print the `n_feats` with the highest sample score for each class
        """

        feat_names = self.features_name
        classes = np.unique(self.Y)

        n_best_feat = n_feats

        df_dict = {}
        for idx, c in enumerate(classes):
            # print(f"\n### Class: {c} ###")
            feats_score = self.sample_weight[idx, :]

            names_scores = list(zip(feat_names, feats_score))
            feat_scores_df = pd.DataFrame(data=names_scores, columns=['Feat_names', 'Score'])
            feat_scores_df = feat_scores_df.sort_values(by=['Score'], ascending=False).reset_index(drop=True)
            df_dict[c] = feat_scores_df

            # best_feats_idx = np.argsort(feats_score)[-n_best_feat:][::-1]
            # best_feats = feat_names[best_feats_idx]

            # print(f"\tBest scores: {best_feats}")

        if to_excel:
            combined_df = pd.concat(df_dict, axis=1)
            combined_df.to_excel(f'MP2/datasets/{self.name}_scores.xlsx')
