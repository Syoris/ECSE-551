import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Data:
    def __init__(self, file, train=False):
        # Download the csv data
        self.file: str = file
        self.train: bool = train

        self.data: pd.DataFrame = self.readData(self.file)

        # Extract the subsets
        self.data_list: list = self.data['body'].to_list()  # List of all samples

        if self.train:
            self.labels = self.data['label'].to_numpy()  # Numpy array with label of each sample

        else:
            self.labels = None

    # Read the data
    def readData(self, file):
        if self.train:
            load_data = pd.read_csv(file, header=None, encoding='latin-1', skiprows=[0], names=['body', 'label'])

        else:
            load_data = pd.read_csv(file, header=None, encoding='latin-1', skiprows=[0], names=['id', 'body'])

        return load_data


class Format_data:
    def __init__(self, words_to_train: Data, max_feat: int=3000, dataset_name:str = 'NoName'):
        self.words_to_train = words_to_train

        self.name: str = dataset_name

        self.max_feat = max_feat

        self.data_text = words_to_train.data_list  # word bank
        self.Y = words_to_train.labels

        self.stop_words = self.stopwords()

        self.data_text = self.process_()  # All samples in text format. (renamed from features)

        self.X, self.vectorizer = self.count_vectorizer()  # vectorizer: To transform text to a vector

        self.features_name = self.vectorizer.get_feature_names_out()  # Corresponding features of vectorizer

    def count_vectorizer(self):
        """
        Create a dictionary of all words.

        To get the CountVectorizer for the training dataset.

        Returns:
            vectorizer and vectorized dataset

        """
        vectorizer = CountVectorizer(stop_words=self.stop_words,
                                     max_features=self.max_feat,
                                     binary=True, )

        # Learn the vocabulary dictionnary and return document term matrix
        X = vectorizer.fit_transform(self.data_text)

        # print(vectorizer.get_feature_names_out())
        X_bin = X.toarray()  # TODO: Check which format is more efficient

        return X_bin, vectorizer

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