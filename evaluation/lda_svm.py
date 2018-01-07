"""
The :mod: `lda_svm` implements the model and the constants needed
for the evalutation of LDA as dimensionality reduction and SVM as
classifier"""
# Author: Ingo Guehring

# import numpy as np
# from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsRestClassifier

import evaluation.shared as shared
import model

# pre_reduction = TruncatedSVD(n_components=500)
PRE_REDUCTION = SparseRandomProjection(n_components=500)
CLASSIFIER = OneVsRestClassifier(SVC(probability=True))

# grid
N_COMPONENTS_RANGE = [1, 2, 4, 6, 8, 10, 12, 13]
# kernels = ['linear', 'rbf']

# old range, that turned out to be too small
# GAMMA_RANGE = np.logspace(-3, 3, 7)
# C_RANGE = np.logspace(-3, 3, 7)

# new wider range
C_RANGE = shared.C_RANGE
GAMMA_RANGE = shared.GAMMA_RANGE

# this could also be used: classifier_kernel=kernels,
PARAM_GRID_DIM = [dict(reduce_dim__n_components=N_COMPONENTS_RANGE,
                       classifier__estimator__gamma=GAMMA_RANGE,
                       classifier__estimator__C=C_RANGE)]

PARAM_GRID = [dict(classifier__estimator__gamma=GAMMA_RANGE,
                   classifier__estimator__C=C_RANGE)]

# model for use in train_apply_classifier
MODEL = model.SMSGuruModel(classifier=CLASSIFIER,
                           pre_reduction=PRE_REDUCTION,
                           reduction=LDA())


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

    Returns
    ---------
    NOTHING but SAVES the results of the performed computations
    """

    MODEL = model.SMSGuruModel(classifier=CLASSIFIER,
                               pre_reduction=PRE_REDUCTION,
                               reduction=LDA(),
                               memory=memory)
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    if gridsearch:
        MODEL.gridsearch(param_grid=PARAM_GRID_DIM,
                         n_jobs=shared.N_JOBS,
                         CV=shared.CV)
        shared.save_and_report(
            results=MODEL.grid_search_.cv_results_,
            folder='lda_svm')

    if gen_error:
        # since in this case the higher the dimension the better the estimator
        # we do not include the lower dimensions in this search
        nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='lda_svm',
                               name='gen_error.npy')
