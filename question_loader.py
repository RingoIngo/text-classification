"""
The :mod:`question_loader` module implements a class which handles the loading
and the properties of the question files
"""
# Author: Ingo Gühring

import csv
import pprint
from datetime import datetime
import operator
import codecs
import numpy as np


class QuestionLoader(object):
    """A loader and container for all properties of the data files.

    Loads all data files and stores the data.
    The data files must have a specific format.

    An example on how to use the class can be found
    in the extraxt_features module.

    Parameters
    ----------
    qfile : csv file containing the questions and the
        corresponding categories. By default the file
        'question_train.csv' is loaded.

    catfile : csv file containing the categoires of the questions.
        a category has a category_nam (string), a category_id (int)
        and a parent_id (int). if the parent_id eqauals zero the
        category has no parent.

    metadata : boolean, determines if also the creation date of the sample
        is extracted. Using the metadata option may reduce the
        number of samples that can be used, since they must
        have a valid date format in the `created_at` section.
        This also means that the resulting classifier can only be
        used on samples with valid date format. In case of the default sample
        data this is a reduction of 36 samples. Dafault is False.

    subcats : boolean which determines if the in qfile specified
        subcategory or its parent category is assigned to a question.

    verbose : boolean, determines if information about the loading process is
        printed to the console.
    """

    def __init__(self, qfile='question_train.csv',
                 catfile='category.csv',
                 metadata=False,
                 subcats=True,
                 verbose=False):
        self.qfile = qfile
        self.catfile = catfile
        self.metadata = metadata
        self.subcats = subcats
        self.categories, self.parentdictionary = self._read_category_file(
            verbose)
        self.questions, self.categoryids = self._read_question_file(verbose)
        self.category_counts = self._get_category_counts(verbose)
        if verbose and self.metadata:
            print("""Warning: Using the metadata option may reduce the
                  number of samples that can be used, since they must
                  have a valid date format in the `created_at` section.
                  This also means that the resulting classifier can only be
                  used on samples with valid date format. In case of the
                  default sample data this is a reduction of 36 samples.""")

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
        with open(self.catfile, 'rU', encoding='utf8') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',')
            f_header = next(reader)

            category_id_idx = f_header.index("category_id")
            parent_id_idx = f_header.index("parent_id")
            category_name_idx = f_header.index("category_name")

            nsyntax_errors = 0
            for row in reader:
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
                    nsyntax_errors = nsyntax_errors + 1
                    continue
        if verbose:
            print("{} lines in {} not read because of syntax errors".format(
                nsyntax_errors, self.catfile))
        return categories, parentdic

    def _read_question_file(self, verbose):
        """reads and stores the questions and corresponding categories.

        If ``subcats`` attribute is false the categories read from
        the file (which are subcategories) are translated to parent
        categories. For this the ``parentdictionary`` attribute is used.

        Returns
        -------
        questions : list containing the questions

        categoryids : numpy.array containing the categoryids for each question.
            It depends on subcats if the category or the parent category is
            stored.

        """
        with open(self.qfile, 'rU', encoding='utf8') as f:
            reader = csv.reader(f, quotechar='"', delimiter=',')
            f_header = next(reader)

            category_main_id_idx = f_header.index("category_main_id")
            question_idx = f_header.index("question")
            created_at_idx = f_header.index("created_at")

            questions = []
            categoryids = []
            nsyntax_errors = 0
            for row in reader:
                try:
                    category_main_id = int(row[category_main_id_idx])
                    question = row[question_idx]
                    if self.metadata:
                        created_at = datetime.strptime(row[created_at_idx],
                                                       "%Y-%m-%d %H:%M:%S")
                    # the model expects a date key even if not used
                    else:
                        created_at = None

                    data = {'question': question, 'created_at': created_at}
                    questions.append(data)
                    if self.subcats:
                        categoryids.append(category_main_id)
                    else:
                        categoryids.append(
                            self.parentdictionary[category_main_id])
                except (ValueError, IndexError):
                    nsyntax_errors = nsyntax_errors + 1
                    continue
        if verbose:
            print("{} not in {} read because of syntax errors".format(
                nsyntax_errors, self.qfile))
        return questions, np.asarray(categoryids)

    def _get_category_counts(self, verbose):
        unique, counts = np.unique(self.categoryids, return_counts=True)
        unique = [(cat_id, self.categories[cat_id]) for cat_id in unique]
        category_counts = dict(zip(unique, counts))
        if verbose:
            print("category counts:")
            pprint.pprint(sorted(category_counts.items(),
                                 key=operator.itemgetter(1), reverse=True))
        return category_counts


if __name__ == "__main__":
    q = QuestionLoader(metadata=True, verbose=True)
    print("nquestions metadata True: {}".format(len(q.questions)))
    q2 = QuestionLoader(metadata=False)
    print("nquestions metadata False: {}".format(len(q2.questions)))
