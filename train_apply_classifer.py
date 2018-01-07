"""
The :mod: `train_apply_classifier` chaines the whole process needed for
predicting a new data set together.
"""
# Author: Ingo Guehring

import argparse
import numpy as np
import evaluation


GRIDSEARCH_MODELS = ['svm-a', 'svm-b', 'knn', 'knn-b']
NO_GRIDSEARCH_MODELS = ['lda']


def train_apply_classifier(classifier='lda',
                           qfile_train='./data/question_train.csv',
                           qcatfile_train='./data/question_category_train.csv',
                           catfile='./data/category.csv',
                           qfile_test='./data/question_test.csv',
                           subcats=False,
                           CV=3,
                           save_results=False,
                           outfile='prediction.npz',
                           verbose=100):

    sms_guru_model, param_grid = evaluation.make_classifier_grid(classifier)
    sms_guru_model.set_question_loader(qfile_train, catfile, subcats)
    sms_guru_model.question_loader_.read_test_data(qfile_test, verbose)

    # load test data
    test_questions = sms_guru_model.question_loader_.test_questions

    if classifier in NO_GRIDSEARCH_MODELS:
        sms_guru_model.fit_transform()
        test_labels = sms_guru_model.predict(test_questions)
    elif classifier in GRIDSEARCH_MODELS:
        # fit all parameters and refit using the best
        sms_guru_model.gridsearch(param_grid, verbose=verbose, CV=CV,
                                  scoring='f1_macro', refit=True)
        # predict with the best estimator
        test_labels = sms_guru_model.grid_search_.predict(test_questions)

    if save_results:
        np.savez(outfile, test_labels=test_labels)
        print(test_labels)
    return test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train apply classifier')
    parser.add_argument('-c', '--classifier',
                        choices=['lda', 'svm-a', 'svm-b', 'knn', 'knn-b'],
                        default=argparse.SUPPRESS)

    parser.add_argument('--save-results', action='store_true',
                        default=argparse.SUPPRESS)

    try:
        train_apply_classifier(**vars(parser.parse_args()))
    except ValueError:
        print("Error")
