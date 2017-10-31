# -*- coding: UTF-8 -*-
# import numpy as np
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import csv
from correctors import SpellingCorrector

SPECIAL_CHARACTERS = [(u'Ä', 'Ae'), ('ä', 'ae'),
                      (u'Ö', 'Oe'), (u'ö', 'oe'),
                      (u'Ü', 'Ue'), (u'ü', 'ue'),
                      (u'ß', 'ss')]


def extract_features(qfile='question_train.csv',
                     qcatfile='question_category_train.csv',
                     catfile='category.csv',
                     spellcorrection=True,
                     stemming=True,
                     subcats=True,
                     tokenization='word_tokenize',
                     outfile='features.npz'):

    categories, parentdic = _readcat(catfile, subcats)
    questions = _readquestion(qfile, subcats, parentdic)
#    print(categories)
#    print(parentdic)
#    print(questions[0])
    if tokenization == 'word_punct_tokenize':
        tokenizer = WordPunctTokenizer()
    else:
        # adviced tokenizer TODO: provide reference
        tokenizer = TreebankWordTokenizer()
    spellcorrector = SpellingCorrector() if spellcorrection else None
    stemmer = SnowballStemmer('german') if stemming else None
    # clean questions
    token_filter = _make_stopwords_and_punct_filter(
        set(stopwords.words('german')))
    normQuestions = []
    for i, (question, cat) in enumerate(questions):
        clean_question = tokenize_and_clean(question,
                                            spellcorrector, stemmer,
                                            token_filter,
                                            tokenizer)
        # add tokens to dictionary set??
        normQuestions.append((clean_question, cat))
        if i < 100:
            print(question)
            print(clean_question)
    # create dictionary from questions
    # TODO

    # extract features from cleaned questions
    # TODO
    return None


def tokenize_and_clean(question, spellcorrector,
                       stemmer, token_filter, tokenizer):
    # TODO is language='german' a valid argument?
    tokens = tokenizer.tokenize(question)
    if spellcorrector is not None:
        # spellcorrect only tokens that are not filtered
        # since stopwords are in small case we filter .lower()
        tokens = [spellcorrector.correct(w) for w in tokens if
                  token_filter(w.lower())]
    # normalize tokens
    # tokens = [lambda w: _normalize(w, spellcorrected) for w in tokens if
    #           token_filter(w.lower())]
    tokens = [_normalize(w, spellcorrector is not None) for w in tokens if
              token_filter(w.lower())]
    # stemm tokens
    if stemmer is not None:
        tokens = [stemmer.stem(w) for w in tokens]
    return tokens


def _make_stopwords_and_punct_filter(stopWords):
    def remove_stop_words_and_punctuation(word):
        return ((word not in stopWords)
                and not all(c in string.punctuation for c in word))
    return remove_stop_words_and_punctuation


def _normalize(word, spellcorrected):
    word = word.lower()
    # if tokens are not spellcorrected, unify special character
    if not spellcorrected:
        for (Umlaut, replacement) in SPECIAL_CHARACTERS:
            word = word.replace(Umlaut, replacement)
    # TODO: deal with numbers and dates
    return word


def _readcat(catfile, subcats):
    categories = {}
    parentdic = {}
    with open(catfile, 'rU') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',')
        f_header = next(reader)
        category_id_idx = f_header.index("category_id")
        parent_id_idx = f_header.index("parent_id")
        category_name_idx = f_header.index("category_name")
        for rowno, row in enumerate(reader):
            try:
                category_id = int(row[category_id_idx])
                parent_id = int(row[parent_id_idx])
                category_name = row[category_name_idx]
                parentdic[category_id] = parent_id
                if ((subcats and parent_id != 0)
                        or (not subcats and parent_id == 0)):
                    categories[category_id] = category_name
                    print("added: ", category_id, parent_id, category_name)
                else:
                    print("neglected: ", category_id, parent_id, category_name)
                    continue
            except (ValueError, IndexError):
                print("Line {} : Syntax error".format(rowno + 1))
                continue
    return categories, parentdic


def _readquestion(qfile, subcats, parentdic):
    with open(qfile, 'rU') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',')
        questions = []
        for rowno, row in enumerate(reader):
            # TODO: don't hardcode position
            # TODO: also correct in other in other functions
            try:
                category_main_id = int(row[3])
                question = row[4]
                if subcats:
                    questions.append((question, category_main_id))
                else:
                    questions.append((question, parentdic[category_main_id]))
            except (ValueError, IndexError):
                print("Line {} : Syntax error".format(rowno + 1))
                continue
    return questions


# run extract_features method if module is executed as a script
if __name__ == "__main__":
    # TODO: make it possible to run it with arguments from command line
    # import sys
    # extract_features(sys.argv[1])
    extract_features(subcats=True)
