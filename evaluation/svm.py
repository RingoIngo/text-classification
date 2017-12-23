"""
The :mod: `svm` implements the model and the constants needed
for the evalutation of SVM as classifier"""
# Author: Ingo Guehring

# import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.random_projection import SparseRandomProjection

import evaluation.shared as shared
import model


reduction = SparseRandomProjection(n_components=500)
CLASSIFIER = OneVsRestClassifier(SVC())
MODEL = model.SMSGuruModel(classifier=CLASSIFIER, reduction=reduction,
                           memory=True, binarize=True)

# grid
# kernels = ['linear', 'rbf']
# finer grid than in lda_svm since less combinations
# GAMMA_RANGE = np.logspace(-3, 3, 10)
# C_RANGE = np.logspace(-3, 3, 10)

# new wider range
C_RANGE = shared.C_RANGE
GAMMA_RANGE = shared.GAMMA_RANGE

PARAM_GRID = [dict(classifier__estimator__gamma=GAMMA_RANGE,
                   classifier__estimator__C=C_RANGE)]


def evaluate(gridsearch=True, gen_error=True):
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    if gridsearch:
        MODEL.gridsearch(param_grid=PARAM_GRID, n_jobs=shared.N_JOBS,
                         CV=shared.CV)
        shared.save_and_report(
            results=MODEL.grid_search_.cv_results_,
            folder='svm')

    if gen_error:
        nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='svm',
                               name='gen_error.npy')
