"""
The :mod: `lda` implements the model and the constants needed
for the evalutation of LDA as classifier"""
# Author: Ingo Guehring

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import TfidfTransformer

import evaluation.shared as shared
import model


MODEL = model.SMSGuruModel(classifier=LDA(), reduction=None, memory=True)

# PARAM_GRID = {}
# PARAM_GRID = dict(classifier__solver=['svd'])
PARAM_GRID = {'union__bow__vectorize__min_df': shared.MIN_DF,
              'union__bow__tfidf': [None, TfidfTransformer()]}


def evaluate(gridsearch=True, gen_error=True):
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
