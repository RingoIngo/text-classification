# import numpy as np
import csv


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
    return None


def _readcat(catfile, subcats):
    with open(catfile, 'rU') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',')
        categories = {}
        parentdic = {}
        for rowno, row in enumerate(reader):
            try:
                category_id = int(row[0])
                parent_id = int(row[1])
                category_name = row[2]
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
