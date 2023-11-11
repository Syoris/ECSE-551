import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
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
    def __init__(self, file, train=False):
        # Download the csv data
        self.file: str = file
        self.train: bool = train

        self.data: pd.DataFrame = self.readData(self.file)
        self._detect_lang()

        # Extract the subsets
        self.data_list: list = self.data['body'].to_list()  # List of all samples

        if self.train:
            self.labels = self.data['label'].to_numpy()  # Numpy array with label of each sample

        else:
            self.labels = None

    # Read the data
    def readData(self, file):
        if self.train:
            load_data = pd.read_csv(file, header=None, encoding='utf-8', skiprows=[0], names=['body', 'label'])

        else:
            load_data = pd.read_csv(file, header=None, encoding='utf-8', skiprows=[0], names=['id', 'body'])

        return load_data

    def _detect_lang(self):
        """
        To find the language (fr or en) of each post
        """
        self.data['lang'] = self.data['body'].apply(lambda x: lang_identifier.classify(x)[0])

class Format_data:
    def __init__(self, words_to_train: Data,
                 max_feat: int | None = 3000,
                 dataset_name:str = 'NoName',
                 n_gram: tuple = (1, 1),
                 binary_features: bool = True, # If true, features are binary, false features are the frequency
                 use_tf_idf: bool = False,
                 pca_n_components: int = 1, # If >1, uses PCA
                 lemmatize: bool = False,
                 ):
        # TODO: Add options
        #   - Lemmatiztion
        #   - language...?

        self.words_to_train = words_to_train
        self.name: str = dataset_name
        self.max_feat = max_feat
        self._n_gram = n_gram
        self._binary_features = binary_features
        self._use_tf_idf = use_tf_idf
        self._pca_n_components = pca_n_components
        self._lemmatize = lemmatize

        self.data_text = words_to_train.data_list  # word bank
        self.Y = words_to_train.labels

        self.stop_words = self.stopwords()
        self._detect_lang()

        self.data_text = self.process_()  # All samples in text format. (renamed from features)


        self.X, self.vectorizer = self.count_vectorizer()  # vectorizer: To transform text to a vector

        self._feature_selection()

        self.features_name = self.vectorizer.get_feature_names_out()  # Corresponding features of vectorizer

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
        else:
            tokenizer = None


        if not self._use_tf_idf:
            vectorizer = CountVectorizer(stop_words=self.stop_words,
                                         max_features=self.max_feat,
                                         ngram_range=self._n_gram,
                                         binary=self._binary_features,
                                         tokenizer=tokenizer
                                         )

        else:
            vectorizer = TfidfVectorizer(stop_words=self.stop_words,
                                         max_features=self.max_feat,
                                         ngram_range=self._n_gram,
                                         binary=False,
                                         tokenizer=tokenizer
                                         )

        # Learn the vocabulary dictionary and return document term matrix
        X = vectorizer.fit_transform(self.data_text)

        # print(vectorizer.get_feature_names_out())
        # X_bin = X.toarray()  # TODO: Check which format is more efficient

        return X, vectorizer

    def process_(self):
        new_data = self.punctuation()
        return new_data

    # Specify stopwords
    def stopwords(self):
        my_stop_words = stopwords.words('english') + stopwords.words('french')
        # print(my_stop_words)
        return my_stop_words

    # Remove punctuations from the dataset
    def punctuation(self):
        punctuations = "?:.,;!"
        for word in self.data_text:
            if word in punctuations:
                self.data_text.remove(word)

        return self.data_text

    # Remove punctuations from the dataset
    def stemming(self):
        punctuations = "?:.,;!"
        for word in self.data_text:
            if word in punctuations:
                self.data_text.remove(word)

        return self.data_text

    def _feature_selection(self):
        """
        To perform PCA analysis
        """
        if self._pca_n_components == 1:
            return

        pca = PCA(n_components=None)

        self.X = pca.fit_transform(self.X)

        plot = False
        if plot:
            sing_values = pca.singular_values_

            # Plot the singular values
            plt.plot(np.arange(1, len(sing_values) + 1), sing_values, marker='o')
            plt.title('Singular Values')
            plt.xlabel('Principal Components')
            plt.ylabel('Singular Values')
            plt.grid(True)
            plt.show()

            ...

