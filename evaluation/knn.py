"""
The :mod: `knn` implements the model and the constants needed
for the evalutation of the kNN classifier"""
# Author: Ingo Guehring

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import evaluation.shared as shared
import model
import model.knn


# algorithm='brute' see stackoverflow --> Zotero, since X is sparse,
# which is good for metric (no real metric)
CLASSIFIER = KNeighborsClassifier(weights=model.knn.cosine_dist_to_sim,
                                  metric=model.knn.cosine_semi_metric,
                                  n_jobs=shared.N_JOBS)
# metadata=False since cosine similiarity measure
MODEL = model.SMSGuruModel(classifier=CLASSIFIER, reduction=None,
                           metadata=False, memory=True)


# grid
N_NEIGHBORS_RANGE = np.arange(5, 65, 5)

PARAM_GRID = dict(classifier__n_neighbors=N_NEIGHBORS_RANGE)


def evaluate():
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    MODEL.gridsearch(param_grid=PARAM_GRID, n_jobs=shared.N_JOBS, CV=shared.CV)
    shared.save_and_report(
        results=MODEL.grid_search_.cv_results_,
        folder='knn')

    nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
    shared.save_and_report(results=nested_scores,
                           folder='knn',
                           name='gen_error.npy')
