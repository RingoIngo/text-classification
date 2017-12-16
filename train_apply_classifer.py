"""
The :mod: `train_apply_classifier` chaines the whole process needed for
predicting a new data set together.
"""
# Author: Ingo Gühring

import evaluation


def train_apply_classifer(classifier='lda_svm',
                          qfile_train='/data/question_train.csv',
                          qcatfile_train='/data/question_category_train.csv',
                          catfile='/data/category.csv',
                          qfile_test='/data/question_test.csv',
                          subcats=False,
                          CV=10,
                          verbose=100):

    sms_guru_model, param_grid = evaluation.make_model(classifier)
    sms_guru_model.set_question_loader(qfile_train, catfile, subcats)
    # fit all parameters and refit using the best
    sms_guru_model.gridsearch(param_grid, verbose=verbose, CV=CV,
                              scoring='f1_micro', refit=True)
    # predict with the best estimator
    return sms_guru_model.grid_search_.predict(qfile_test)
