"""
The :mod: `text_tokenizer_and_cleaner` contains the `TextTokenizerAndCleaner`
class, which handles the preprocessing of the text
"""
# Author: Ingo Gühring
# This module is inspired by the book
#    `Applied Text Analysis with Python` by
#    Bengfort, Ojeda and Bilbro

import string
from datetime import datetime

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.base import BaseEstimator, TransformerMixin

from model.text.correctors import SpellingCorrector


class TextTokenizerAndCleaner(BaseEstimator, TransformerMixin):
    """A text preprocessor.

    Tokenizes and cleans text. Cleaning may consist of
    spell correction, a mapping of dates, times and other
    numeric data, removing of stopwords, converting to
    lower case and stemming. Default values are justified
    by the evaluation in milestone 1 (see report and presentation).

    Because of its interfaces the class can be used in
    a `Pipeline`

    Parameters
    ----------
    mapdates : boolean, if True all dates are mapped to a dummy date
        Default is True.

    mapnumbers : boolean, if True and `mapdates` also True,
        then all tokens which are not a date and contain a
        number are mapped to a dummy number. If True and
        `mapdates` is False, then all tokens containing a
        number are mapped to a dummy number. Default is False.

    spellcorrect : boolean, if True tokens will be spell
        corrected. Since this is very time consuming default
        is False.

    stem : boolean, if True the Snowball stemmer for
        the german language will be applied to all tokens.
        Default is True.

    tokenizer : string, either `word_tokenizer` or
        `word_punct_tokenizer`. If `word_tokenizer`
        TreebankWordTokenizer from nltk package is used.
        This is the from nltk adviced tokenizer
        (see: http://www.nltk.org/api/nltk.tokenize.html).
        If `word_punct_tokenizer` the `WordPunctTokenizer` from nltk
        package is used, which is also the defualt value.
        If any other string is given the default
        value is applied.
    """

    special_characters_map = [(u'Ä', 'Ae'), ('ä', 'ae'),
                              (u'Ö', 'Oe'), (u'ö', 'oe'),
                              (u'Ü', 'Ue'), (u'ü', 'ue'),
                              (u'ß', 'ss')]
    date_patterns = ["%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%y-%m-%d",
                     "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%y/%m/%d",
                     "%d.%m.%Y", "%d.%m.%y", "%Y.%m.%d", "%y.%m.%d",
                     "%Y"]

    def __init__(self, mapdates=True, mapnumbers=False, spellcorrect=False,
                 stem=True, tokenizer='word_punct_tokenizer'):
        self.spellcorrect = spellcorrect
        self.stem = stem
        self.tokenizer = tokenizer
        self.mapdates = mapdates
        self.mapnumbers = mapnumbers

    def _set_tokenizer(self):
        """Set tokenizer to either `WordPunctTokenizer` or
        `TreebankTokenizer`
        """
        if self.tokenizer == 'word_punct_tokenizer':
            self._tokenizer = WordPunctTokenizer()
        elif self.tokenizer == 'word_tokenizer':
            # TODO: there sould be an option to turn off
            # escapeForwardSlashAsterisk, which produces
            # weird behaviour
            self._tokenizer = TreebankWordTokenizer()
        else:
            raise ValueError(""""tokenizer must be either word_punct_tokenizer
                             or word_tokenizer""")

    def is_punct(self, token):
        """Return is token consists only of punctuation"""
        return all(c in string.punctuation for c in token)

    def is_stopword(self, token):
        """Return if token is stopword"""
        return token.lower() in self.stopwords

    def is_pattern(self, token, patterns):
        """Return if token corresponds to a pattern in patterns"""
        for pattern in patterns:
            try:
                # print(datetime.strptime(token, pattern))
                datetime.strptime(token, pattern)
                return True
            except ValueError:
                pass
        return False

    def has_number(self, token):
        return any(char.isdigit() for char in token)

    def is_time(self, token):
        """"Return if token is time"""
        return False

    def map_special_char(self, token):
        """Replace german Umlauts with alternative characters and
        return token
        """
        for (Umlaut, replacement) in self.special_characters_map:
            token = token.replace(Umlaut, replacement)
        return token

    def filter_token(self, token):
        """Return True if token is not a stopword and not only
        punctuation
        """
        return not self.is_stopword(token.lower()) and not self.is_punct(token)

    def normalize(self, token):
        """Apply specified text preprocessing methods to token
        and return token
        """
        token = token.lower()
        # if tokens are not spellcorrected, unify special character
        if self.spellcorrect:
            token = self.map_special_char(token)
        if self.mapdates and self.is_pattern(token, self.date_patterns):
            token = "00.00.00"
        elif self.mapnumbers and self.has_number(token):
            token = "0"

        return token

    def tokenize_and_clean(self, question):
        """Apply specified text preprocessing methods on question
        and return list of cleaned tokens
        """
        tokens = self._tokenizer.tokenize(question)
        if self.spellcorrect:
            # spellcorrect only tokens that are not filtered
            tokens = [self.spellcorrector.correct(token) for token in tokens if
                      self.filter_token(token)]
        # normalize tokens
        tokens = [self.normalize(token) for token in tokens if
                  self.filter_token(token)]
        # stemm tokens
        if self.stem:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    # Estimator interface

    def fit(self, X, y=None):
        """Nothing needs to be learned from data. Return self"""
        return self

    def transform(self, questions):
        self.stopwords = set(stopwords.words('german'))
        # add also 'Umlaut' variants of stopwords to stopwords
        self.stopwords = self.stopwords.union(
            [self.map_special_char(w) for w in self.stopwords])

        self._set_tokenizer()
        if self.spellcorrect:
            self.spellcorrector = SpellingCorrector()

        if self.stem:
            self.stemmer = SnowballStemmer('german')

        """Process each question in the list questions"""
        for question in questions:
            yield self.tokenize_and_clean(question)


# if __name__ == "__main__":
#     from sklearn.utils.estimator_checks import check_estimator
#     check_estimator(TextTokenizerAndCleaner())
