# -*- coding: UTF-8 -*-
import csv
from sklearn.cross_validation import KFold
import numpy as np


class QuestionLoader(object):

    def __init__(self, qfile='question_train.csv',
                 catfile='category.csv',
                 folds=None,
                 shuffle=True,
                 subcats=False):
        self.qfile = qfile
        self.catfile = catfile
        self.subcats = subcats
        self.folds = folds
        self.categories, self.parentdictionary = self._read_category_file()
        self.questions, self.categoryids = self._read_question_file()

    def _read_category_file(self):
        categories = {}
        parentdic = {}
        with open(self.catfile, 'rU') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',')
            f_header = next(reader)
            category_id_idx = f_header.index("category_id")
            parent_id_idx = f_header.index("parent_id")
            category_name_idx = f_header.index("category_name")
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
                        print("added: ", category_id, parent_id, category_name)
                    else:
                        print("neglected: ", category_id, parent_id,
                              category_name)
                        continue
                except (ValueError, IndexError):
                    print("line {} : syntax error".format(rowno + 1))
                    continue
        return categories, parentdic

    def _read_question_file(self):
        with open(self.qfile, 'rU') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',')
            questions = []
            categoryids = []
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
                    print("line {} : syntax error".format(rowno + 1))
                    continue
        return questions, np.array(categoryids)
