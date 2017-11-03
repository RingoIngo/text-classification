# -*- coding: UTF-8 -*-
# import numpy as np
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from correctors import SpellingCorrector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline


def identity(words):
    return words


class TextTokenizerAndCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, mapnumerics=True, spellcorrector=False,
                 stemmer=True, tokenizer='word_tokenizer'):
        self.special_characters_map = [(u'Ä', 'Ae'), ('ä', 'ae'),
                                       (u'Ö', 'Oe'), (u'ö', 'oe'),
                                       (u'Ü', 'Ue'), (u'ü', 'ue'),
                                       (u'ß', 'ss')]
        self.stopwords = set(stopwords.words('german'))
        # add also 'Umlaut' variants of stopwords to stopwords
        self.stopwords = self.stopwords.union(
            [self.map_special_char(w) for w in self.stopwords])
        # note getter and setter methods
        self.spellcorrector = spellcorrector
        self.stemmer = stemmer
        self.tokenizer = tokenizer
        self.mapnumerics = mapnumerics

    @property
    def spellcorrector(self):
        return self.__spellcorrector

    @spellcorrector.setter
    def spellcorrector(self, spellcorrector):
        self.__spellcorrector = SpellingCorrector() if spellcorrector else None

    @property
    def stemmer(self):
        return self.__stemmer

    @stemmer.setter
    def stemmer(self, stemmer):
        self.__stemmer = SnowballStemmer('german') if stemmer else None

    @property
    def tokenizer(self):
        return self.__tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        if tokenizer == 'word_punct_tokenizer':
            self.__tokenizer = WordPunctTokenizer()
        else:
            # adviced tokenizer TODO: provide reference
            self.__tokenizer = TreebankWordTokenizer()

    def is_punct(self, token):
        return all(c in string.punctuation for c in token)

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def is_date(self, token):
        return False

    def is_time(self, token):
        return False

    def map_special_char(self, token):
        for (Umlaut, replacement) in self.special_characters_map:
            token = token.replace(Umlaut, replacement)
        return token

    def filter_token(self, token):
        # TODO maybe should depend on input
        return not self.is_stopword(token.lower()) and not self.is_punct(token)

    def normalize(self, token):
        token = token.lower()
        # if tokens are not spellcorrected, unify special character
        if self.spellcorrector is None:
            token = self.map_special_char(token)
        # TODO: deal with numbers and dates
        if self.mapnumerics:
            token = token
        return token

    def tokenize_and_clean(self, question):
        # TODO is language='german' a valid argument?
        tokens = self.tokenizer.tokenize(question)
        if self.spellcorrector is not None:
            # spellcorrect only tokens that are not filtered
            tokens = [self.spellcorrector.correct(token) for token in tokens if
                      self.filter_token(token)]
        # normalize tokens
        tokens = [self.normalize(token) for token in tokens if
                  self.filter_token(token)]
        # stemm tokens
        if self.stemmer is not None:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def fit(self, X, y=None):
        return self

    def transform(self, questions):
        for question in questions:
            yield self.tokenize_and_clean(question)


def create_pipeline(estimator=None):

    steps = [
        ('tokens', TextTokenizerAndCleaner()),
        ('vectorize', CountVectorizer(tokenizer=identity, preprocessor=None,
                                      lowercase=False)),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('reduction', TruncatedSVD(n_components=10000))
    ]

    if estimator is not None:
        # Add the estimator
        steps.append(('classifier', estimator))
    return Pipeline(steps)
