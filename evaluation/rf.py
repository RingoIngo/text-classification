"""
The :mod: `rf` implements the model and the constants needed
for the evalutation of the random forest classifier"""
# Author: Ingo Guehring

# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import evaluation.shared as shared
import model


# CLASSIFIER = RandomForestClassifier(n_estimators=500)
CLASSIFIER = GradientBoostingClassifier(n_estimators=500)

# MAX_FEATURES_RANGE = [.2, .4, .6, .8, 'log2', 'sqrt']
MAX_FEATURES_RANGE = ['log2']


PARAM_GRID = [dict(classifier__max_features=MAX_FEATURES_RANGE)]

# model for use in train_apply_classifier
MODEL = model.SMSGuruModel(classifier=CLASSIFIER, metadata=False)


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
    MODEL = model.SMSGuruModel(classifier=CLASSIFIER, memory=memory,
                               metadata=False)
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    if gridsearch:
        MODEL.gridsearch(param_grid=PARAM_GRID, n_jobs=shared.N_JOBS,
                         CV=shared.CV)
        shared.save_and_report(
            results=MODEL.grid_search_.cv_results_,
            folder='rf')

    if gen_error:
        nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='rf',
                               name='gen_error.npy')
