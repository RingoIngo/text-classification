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

# grid
N_NEIGHBORS_RANGE = np.arange(5, 65, 5)

PARAM_GRID = dict(classifier__n_neighbors=N_NEIGHBORS_RANGE)
MODEL = model.SMSGuruModel(classifier=CLASSIFIER, reduction=None,
                           metadata=False)


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

    memory : boolean, if True memory option is used

    Returns
    ---------
    NOTHING but SAVES the results of the performed computations
    """
    # metadata=False since cosine similiarity measure
    MODEL = model.SMSGuruModel(classifier=CLASSIFIER, reduction=None,
                               metadata=False, memory=memory)
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    if gridsearch:
        MODEL.gridsearch(param_grid=PARAM_GRID, n_jobs=shared.N_JOBS,
                         CV=shared.CV)
        shared.save_and_report(
            results=MODEL.grid_search_.cv_results_,
            folder='knn')

    if gen_error:
        nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='knn',
                               name='gen_error.npy')
