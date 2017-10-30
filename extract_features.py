#-*- coding: UTF-8 -*-
# import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
import nltk

SPECIAL_CHARACTERS = [(u'Ä', 'Ae'), ('ä', 'ae'),
                      (u'Ö', 'Oe'), (u'ö', 'oe'),
                      (u'Ü', 'Ue'), (u'ü', 'ue'),
                      (u'ß', 'ss')]


def extract_features(qfile='question_train.csv',
                     qcatfile='question_category_train.csv',
                     catfile='category.csv',
                     subcats=True,
                     outfile='features.npz'):

    categories, parentdic = _readcat(catfile, subcats)
    questions = _readquestion(qfile, subcats, parentdic)
#    print(categories)
#    print(parentdic)
#    print(questions[0])
    stopWords = set(stopwords.words('german'))
    normQuestions = []
    for (question, cat) in questions:
        print(question)
        # print(nltk.tokenize.wordpunct_tokenize(q[0]))
        # adviced tokenizer
        tokens = word_tokenize(question, language='german')
        # normalize tokens
        # TODO: normalize, i.e. small case, spell check, map numebers and dates
        normtokens = [_normalize(z) for z in tokens if z.lower() not in stopWords]
        cleanNormTokens = list(filter(
            lambda w: _removeStopWordsAndPunc(w, stopWords), normtokens))
        normQuestions.append((cleanNormTokens, cat))
        print(cleanNormTokens)
    return None


def _removeStopWordsAndPunc(word, stopWords):
    return ((word not in stopWords)
            and not (all(c in string.punctuation for c in word)))


def _normalize(word):
    word = word.lower()
    for (Umlaut, replacement) in SPECIAL_CHARACTERS:
        word = word.replace(Umlaut, replacement)
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
