"""
The :mod: `lda` implements the model and the constants needed
for the evalutation of LDA as classifier"""
# Author: Ingo Guehring

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import evaluation.shared as shared
import model


MODEL = model.SMSGuruModel(classifier=LDA(), reduction=None, memory=True)

# PARAM_GRID = {}
PARAM_GRID = dict(classifier__solver=['svd'])


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
            folder='lda')

    if gen_error:
        nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='lda',
                               name='gen_error.npy')
