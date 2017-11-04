"""
The :mod:`question_loade` module implements a class which handles the loading
and the properties of the question files
"""
# Author: Ingo Gühring

import csv
import numpy as np

from sklearn.cross_validation import KFold


class QuestionLoader(object):
    """A loader and container for all properties of the data files.

    Load all data files and store the data.
    The data files must have a specific format.

    An example on how to use the class can be found
    in the extraxt_features module.

    Parameters
    ----------
    qfile : csv file containing the questions and the
        corresponding categories. By defailt the file
        'question_train.csv' is loaded.

    catfile : csv file containing the categoires of the questions.
        A category has a category_nam (string), a category_id (int)
        and a parent_id (int). If the parent_id eqauals zero the
        category has no parent.

    folds : TODO

    shuffle : TODO

    subcats : boolean which determines if the in qfile specified
        subcategory or its parent category is assigned to a question.

    verbose : boolean, determines if information about the loading process is
        printed to the console.
    """

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
        """read categoriy_id, parent_id and category_name from file

        Reads data from files and builds a dictionary which stores for each
        category the id of its parent category.

        Returns
        -------
        categories : dictionary storing the category name under the category id

        parentdic : dictionary which stores for each category the id of its
            parent category. If the category has no parent category ``0`` is
            stored.
        """
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
        """reads and stores the questions and corresponding categories.

        Returns
        -------
        questions : list containing the questions

        categoryids : numpy.array containing the categoryids for each question.
            It depends on subcats if the category or the parent category is
            stored.

        """
        with open(self.qfile, 'rU') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',')
            f_header = next(reader)
            category_main_id_idx = f_header.index("category_main_id")
            question_idx = f_header.index("question")
            questions = []
            categoryids = []
            nSyntaxErrors = 0
            for rowno, row in enumerate(reader):
                try:
                    category_main_id = int(row[category_main_id_idx])
                    question = row[question_idx]
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
