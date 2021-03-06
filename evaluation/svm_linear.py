"""
The :mod: `svm_linear` implements the model and the constants needed
for the evalutation of SVM as classifier with a linear kernel"""
# Author: Ingo Guehring

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

import evaluation.shared as shared
import model


CLASSIFIER = LinearSVC()
# CLASSIFIER = BaggingClassifier(LinearSVC(), n_estimators=5, max_samples=0.5)

# new wider range
# C_RANGE = shared.C_RANGE
C_RANGE = np.logspace(-5, 5, 11)

PARAM_GRID = {'classifier__base_estimator__C': C_RANGE,
              'union__bow__vectorize__min_df': shared.MIN_DF,
              'union__bow__tfidf': [None, TfidfTransformer()]}

# model for use in train_apply_classifier
MODEL = model.SMSGuruModel(classifier=CLASSIFIER, reduction=None)


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
    CCV = KFold(n_splits=3) if shared.SUBCATS else 3
    MODEL = model.SMSGuruModel(
        CalibratedClassifierCV(LinearSVC(), cv=CCV), reduction=None,
        memory=memory)
    MODEL.set_question_loader(subcats=shared.SUBCATS)
    if gridsearch:
        MODEL.gridsearch(param_grid=PARAM_GRID, n_jobs=shared.N_JOBS,
                         CV=shared.CV)
        shared.save_and_report(
            results=MODEL.grid_search_.cv_results_,
            folder='svm_linear')

    if gen_error:
        nested_scores = MODEL.nested_cv(param_grid=PARAM_GRID, CV=shared.CV)
        shared.save_and_report(results=nested_scores,
                               folder='svm_linear',
                               name='gen_error.npy')
