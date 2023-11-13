import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
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
    def __init__(self, words_dataset: Data,
                 max_feat: int | None = 3000,
                 dataset_name:str = 'NoName',
                 n_gram: tuple = (1, 1),
                 binary_features: bool = True,  # If true, features are binary, false features are the frequency
                 use_tf_idf: bool = False,
                 pca_n_components: int = 1,  # If >1, uses PCA
                 lemmatize: bool = False,
                 lang_id:bool = False,  # If true, add a feature for the language (0: en, 1:fr)
                 ):

        # Attributes
        self.words_dataset: Data = words_dataset
        self.name: str = dataset_name
        self.max_feat = max_feat
        self._n_gram = n_gram
        self._binary_features = binary_features
        self._use_tf_idf = use_tf_idf
        self._pca_n_components = pca_n_components
        self._lemmatize = lemmatize
        self._lang_id = lang_id

        print(f"\tProcessing of: {self.name}")
        # Train labels
        self.Y = words_dataset.labels
        self.X_test = None

        # Pre-process
        self.train_text, self.test_text = self._pre_process_text()  # All samples in text format. (renamed from features)
        self.stop_words = self.get_stop_words()

        # Tokenize
        self.X, self.vectorizer = self.count_vectorizer()  # vectorizer: To transform text to a vector
        self.pca = None  # PCA transformer

        self._add_lang()
        self._feature_selection()

        self.features_name = self.vectorizer.get_feature_names_out()  # Corresponding features of vectorizer

        file_name = f'MP2/datasets/{self.name}.pkl'
        print(f"Saving to ")

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
        To perform PCA analysis
        """
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

    def _add_lang(self, transform_test=False):
        """
        Add a feature with the language of the post (en:0, fr:1)
        """
        if self._lang_id:
            if not transform_test:
                lang_array = (self.words_dataset.train_data['lang'] == 'fr').astype(int).to_numpy() # 0: en, 1:fr
                lang_sparse = sp.csr_matrix(lang_array).reshape(-1, 1)

                self.X = sp.csr_matrix(sp.hstack([self.X, lang_sparse]))

            else:
                lang_array = (self.words_dataset.test_data['lang'] == 'fr').astype(int).to_numpy() # 0: en, 1:fr
                lang_sparse = sp.csr_matrix(lang_array).reshape(-1, 1)

                self.X_test = sp.csr_matrix(sp.hstack([self.X_test, lang_sparse]))

    def transform_test(self):
        # Tokenize
        self.X_test = self.vectorizer.transform(self.test_text)

        # PCA
        if self.pca is not None:
            if isinstance(self.X, sp.csr_matrix):
                self.X_test = self.X.toarray()
            self.X_test = self.pca.transform(self.X_test)

        # Language
        self._add_lang(transform_test=True)

        return self.X_test
