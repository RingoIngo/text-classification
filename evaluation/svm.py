"""
The :mod: `svm` implements the model and the constants needed
for the evalutation of SVM as classifier"""
# Author: Ingo Guehring

import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import evaluation.shared as shared
import model


CLASSIFIER = OneVsRestClassifier(SVC())
MODEL = model.SMSGuruModel(classifier=CLASSIFIER, reduction=None, memory=True)

# grid
# kernels = ['linear', 'rbf']
# finer grid than in lda_svm since less combinations
GAMMA_RANGE = np.logspace(-3, 3, 10)
C_RANGE = np.logspace(-3, 3, 10)

PARAM_GRID = [dict(classifier__gamma=GAMMA_RANGE, classifier__C=C_RANGE)]


def evaluate():
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    MODEL.gridsearch(param_grid=PARAM_GRID, n_jobs=shared.N_JOBS, CV=shared.CV)
    shared.save_and_report(
        results=MODEL.grid_search_.cv_results_,
        folder='svm')

    nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
    shared.save_and_report(results=nested_scores,
                           folder='svm',
                           name='gen_error.npy')
