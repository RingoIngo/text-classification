"""
The :mod: `lda_svm` implements the model and the constants needed
for the evalutation of LDA as dimensionality reduction and SVM as
classifier"""
# Author: Ingo Guehring

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsRestClassifier

import evaluation.shared as shared
import model

# pre_reduction = TruncatedSVD(n_components=500)
pre_reduction = SparseRandomProjection(n_components=500)
classifier = OneVsRestClassifier(SVC())
# binarize=True would be good for AUC score but LDA takes labels NOT in
# binary format
MODEL = model.SMSGuruModel(classifier=classifier,
                           pre_reduction=pre_reduction,
                           reduction=LDA(),
                           memory=True)

# grid
N_COMPONENTS_RANGE = [1, 2, 4, 6, 8, 10, 12, 14]
# kernels = ['linear', 'rbf']
GAMMA_RANGE = np.logspace(-3, 3, 7)
C_RANGE = np.logspace(-3, 3, 7)

# this could also be used: classifier_kernel=kernels,
PARAM_GRID_DIM = [dict(reduce_dim__n_components=N_COMPONENTS_RANGE,
                       classifier__estimator__gamma=GAMMA_RANGE,
                       classifier__estimator__C=C_RANGE)]

PARAM_GRID = [dict(classifier__estimator__gamma=GAMMA_RANGE,
                   classifier__estimator__C=C_RANGE)]


def evaluate():
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    MODEL.gridsearch(param_grid=PARAM_GRID_DIM,
                     n_jobs=shared.N_JOBS,
                     CV=shared.CV)
    shared.save_and_report(
        results=MODEL.grid_search_.cv_results_,
        folder='lda_svm')

    # since in this case the higher the dimension the better the estimator
    # we do not include the lower dimensions in this search
    nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
    shared.save_and_report(results=nested_scores,
                           folder='lda_svm',
                           name='gen_error.npy')
