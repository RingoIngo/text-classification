"""
The :mod: `multinb` implements the model and the constants needed
for the evalutation of multinomial naive bayes as classifier"""
# Author: Ingo Guehring

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfTransformer

import evaluation.shared as shared
import model


# MODEL = model.SMSGuruModel(classifier=MultinomialNB(), reduction=None,
#                            metadata=False, memory=True)
#
# PARAM_GRID = dict(classifier__alpha=np.array([1]))
MODEL = model.SMSGuruModel(
    CalibratedClassifierCV(MultinomialNB(), method='isotonic'), reduction=None,
    metadata=False, memory=True).model

PARAM_GRID = {'union__bow__vectorize__min_df': shared.MIN_DF,
              'union__bow__tfidf': [None, TfidfTransformer()]}


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
            folder='multinb')

    if gen_error:
        nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='multinb',
                               name='gen_error.npy')
