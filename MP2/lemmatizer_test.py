from nltk.stem import PorterStemmer
import nltk
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

train_corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
     'Is this the first document?',]
test_corpus = ['This is the fourth document.']


tokens = word_tokenize(train_corpus[0])
print(tokens)

class StemTokenizer:
     def __init__(self):
       self.wnl =PorterStemmer()
     def __call__(self, doc):
       return [self.wnl.stem(t) for t in word_tokenize(doc) if t.isalpha()]

new_dset = ['My cars are beautiful', "playing", "he plays football","she is student",'was', "they are students","connect", "connection",'retrieval', 'retrieved', 'retrieves']
vect = CountVectorizer(tokenizer=StemTokenizer())
vect.fit_transform(new_dset)
vect.get_feature_names_out()

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer:
     def __init__(self):
       self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
       return [self.wnl.lemmatize(t,pos ="v") for t in word_tokenize(doc) if t.isalpha()]


from nltk.corpus import wordnet


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()

# 2. Lemmatize Single Word with the appropriate POS tag
word = 'feet'
print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

# 3. Lemmatize a Sentence with the appropriate POS tag
sentence = "The striped bats are hanging on their feet for best"
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])


class New_LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t, pos=get_wordnet_pos(t)) for t in word_tokenize(doc) if t.isalpha()]


new_dset = ['My cars are beautiful', "playing", "he plays football","she is student","was", "they are students","connect", "connection","The striped bats are hanging on their feet for best"]
vect = CountVectorizer(tokenizer=New_LemmaTokenizer())
a = vect.fit_transform(new_dset)
vect.get_feature_names_out()
print(a.shape)