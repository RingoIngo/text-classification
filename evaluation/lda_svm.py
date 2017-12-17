"""
The :mod: `evaluate_lda_svm` implements the model and the constants needed
for the evalutation of LDA as dimensionality reduction and SVM as
classifier"""
# Author: Ingo Gühring

import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import evaluation.shared as shared
import model


MODEL = model.SMSGuruModel(classifier=SVC(), reduction=LDA())

# grid
N_COMPONENTS_RANGE = np.arange(1, shared.N_PARENTCATS + 1)
# kernels = ['linear', 'rbf']
GAMMA_RANGE = np.logspace(-3, 3, 10)
C_RANGE = np.logspace(-3, 3, 10)


def make_grid(dim=False):
    if dim:
        # this could also be used: classifier_kernel=kernels,
        param_grid = [dict(reduce_dim__n_components=N_COMPONENTS_RANGE,
                           classifier__gamma=GAMMA_RANGE,
                           classifier__C=C_RANGE)]
    else:
        param_grid = [dict(classifier__gamma=GAMMA_RANGE,
                           classifier__C=C_RANGE)]
    return param_grid


def evaluate():
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    grid_gridsearch = make_grid(dim=True)
    MODEL.gridsearch(param_grid=grid_gridsearch,
                     n_jobs=shared.N_JOBS,
                     CV=shared.CV)
    shared.save_and_report(
        results=MODEL.grid_search_.cv_results_,
        folder='lda_svm')

    since in this case the higher the dimension the better the estimator
    we do not include the lower dimensions in this search
    grid_generalization_error = make_grid()
    nested_scores = MODEL.nested_cv(
        param_grid=grid_generalization_error, CV=shared.CV)
    shared.save_and_report(results=nested_scores,
                           folder='lda_svm',
                           name='gen_error.npy')
