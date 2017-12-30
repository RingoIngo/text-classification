"""
The :mod: `qda` implements the model and the constants needed
for the evalutation of QDA as classifier"""
# Author: Ingo Guehring

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

import evaluation.shared as shared
import model


MODEL = model.SMSGuruModel(classifier=QDA(), reduction=None, memory=True)

PARAM_GRID = dict(classifier__tol=np.array([10**(-4)]))


def evaluate(gridsearch=True, gen_error=True):
    # since there are no hyper parameters to be optimized we only need
    # the generalization error estimate
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    if gridsearch:
        MODEL.gridsearch(param_grid=PARAM_GRID,
                         n_jobs=shared.N_JOBS,
                         CV=shared.CV)
        shared.save_and_report(
            results=MODEL.grid_search_.cv_results_,
            folder='qda')

    if gen_error:
        nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='qda',
                               name='gen_error.npy')
