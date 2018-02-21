"""
The :mod: `svm_linear` implements the model and the constants needed
for the evalutation of SVM as classifier with a linear kernel"""
# Author: Ingo Guehring

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

import model.question_loader as ql
import evaluation.shared as shared
import model


# CLASSIFIER = LinearSVC()
CLASSIFIER = BaggingClassifier(LinearSVC(C=0.1))

# new wider range
# C_RANGE = shared.C_RANGE
C_RANGE = np.logspace(-5, 5, 11)

PARAM_GRID = [dict(classifier__C=C_RANGE)]

# model for use in train_apply_classifier
MODEL = model.SMSGuruModel(classifier=CLASSIFIER, reduction=None)


def evaluate(gridsearch=True, gen_error=True, memory=True):
    """Evaluate model

    Compute either an estimate for the generalization error for
    f1_macro with a nested gridsearch or evaluate the parameter
    grid in a simple gridsearch.

    Parameters
    -----------
    gridsearch : boolean, if True the gridsearch is performed

    gen_error : boolean, if True an estimate for the generalization
        error is computed.

    memory : boolean, if True memory option is used

    Returns
    ---------
    NOTHING but SAVES the results of the performed computations
    """
    MODEL = model.SMSGuruModel(classifier=CLASSIFIER, reduction=None,
                               memory=memory)
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    if gridsearch:
        MODEL.gridsearch(param_grid=PARAM_GRID, n_jobs=shared.N_JOBS,
                         CV=shared.CV)
        shared.save_and_report(
            results=MODEL.grid_search_.cv_results_,
            folder='svm_linear')

    if gen_error:
        question_loader = ql.QuestionLoader(
            qfile=shared.QFILE, catfile=shared.CATFILE, subcats=False,
            metadata=True, verbose=True)
        nested_scores = cross_val_score(
            MODEL.model, X=question_loader.questions,
            y=question_loader.categoryids, cv=5,
            scoring='f1_macro', verbose=100)
        # nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='svm_linear',
                               name='gen_error_bagging.npy')
