# -*- coding: UTF-8 -*-
# import numpy as np
import csv
from pprint import pprint
import create_model


def extract_features(qfile='question_train.csv',
                     qcatfile='question_category_train.csv',
                     catfile='category.csv',
                     countwords=True,
                     metadata=True,
                     spellcorrection=False,
                     stemming=True,
                     subcats=False,
                     tokenization='word_tokenize',
                     outfile='features.npz'):

    categories, parentdic = _readcat(catfile, subcats)
    questions, qcats = _readquestion(qfile, subcats, parentdic)
    model = create_model.create_pipeline()
    model.set_params(reduction=None)
    features = model.fit_transform(questions, qcats)
    # featurenames = model.vocabulary_
    featurenames = model.named_steps['vectorize'].get_feature_names()
    categoryids = qcats
    print("feature size{}".format(features.shape))
    print("featurenames size {}".format(len(featurenames)))
    print("categoryids size {}".format(len(categoryids)))
    print("categories size: {}".format(len(categories)))
    print("number of questions: {}".format(len(questions)))
    print(categories)
    return None


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
        categories = []
        for rowno, row in enumerate(reader):
            # TODO: don't hardcode position
            # TODO: also correct in other in other functions
            try:
                category_main_id = int(row[3])
                question = row[4]
                questions.append(question)
                if subcats:
                    categories.append(category_main_id)
                else:
                    categories.append(parentdic[category_main_id])
            except (ValueError, IndexError):
                print("Line {} : Syntax error".format(rowno + 1))
                continue
    return questions, categories


# run extract_features method if module is executed as a script
if __name__ == "__main__":
    # TODO: make it possible to run it with arguments from command line
    # import sys
    # extract_features(sys.argv[1])
    extract_features(subcats=True)
