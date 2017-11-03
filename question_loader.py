# -*- coding: UTF-8 -*-
import csv
from sklearn.cross_validation import KFold
import numpy as np


class QuestionLoader(object):

    def __init__(self, qfile='question_train.csv',
                 catfile='category.csv',
                 folds=None,
                 shuffle=True,
                 subcats=True,
                 verbose=False):
        self.qfile = qfile
        self.catfile = catfile
        self.subcats = subcats
        self.folds = folds
        self.categories, self.parentdictionary = self._read_category_file(
            verbose)
        self.questions, self.categoryids = self._read_question_file(verbose)

    def _read_category_file(self, verbose):
        categories = {}
        parentdic = {}
        with open(self.catfile, 'rU') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',')
            f_header = next(reader)
            category_id_idx = f_header.index("category_id")
            parent_id_idx = f_header.index("parent_id")
            category_name_idx = f_header.index("category_name")
            nSyntaxErrors = 0
            for rowno, row in enumerate(reader):
                # filter category "n" and junk
                try:
                    category_id = int(row[category_id_idx])
                    parent_id = int(row[parent_id_idx])
                    category_name = row[category_name_idx]
                    parentdic[category_id] = parent_id
                    if ((self.subcats and parent_id != 0)
                            or (not self.subcats and parent_id == 0)):
                        categories[category_id] = category_name
                        if verbose:
                            print("added category: ", category_id,
                                  parent_id, category_name)
                    else:
                        if verbose:
                            print("neglected category: ", category_id,
                                  parent_id, category_name)
                        continue
                except (ValueError, IndexError):
                    nSyntaxErrors = nSyntaxErrors + 1
                    continue
        if verbose:
            print("{} lines in {} not read because of syntax errors".format(
                nSyntaxErrors, self.catfile))
        return categories, parentdic

    def _read_question_file(self, verbose):
        with open(self.qfile, 'rU') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',')
            questions = []
            categoryids = []
            nSyntaxErrors = 0
            for rowno, row in enumerate(reader):
                # todo: don't hardcode position
                # todo: also correct in other in other functions
                try:
                    category_main_id = int(row[3])
                    question = row[4]
                    questions.append(question)
                    if self.subcats:
                        categoryids.append(category_main_id)
                    else:
                        categoryids.append(
                            self.parentdictionary[category_main_id])
                except (ValueError, IndexError):
                    nSyntaxErrors = nSyntaxErrors + 1
                    continue
        if verbose:
            print("{} not in {} read because of syntax errors".format(
                nSyntaxErrors, self.qfile))
        return questions, np.array(categoryids)
