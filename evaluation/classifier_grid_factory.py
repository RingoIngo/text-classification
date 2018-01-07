"""
The :mod: `classifier_grid_factory` translates the string name of the
classification method into a model and associated parameter grid if
existing.
"""
# Author: Ingo Guehring

import evaluation.lda_svm as lda_svm
import evaluation.svm as svm
import evaluation.lda as lda
import evaluation.knn as knn
import evaluation.knn_b as knn_b


def make_classifier_grid(classifier):
    """Return SMSGuruModel with classification method and associated
    parameter grid.

    Parameters
    ----------

    classifier : string, name of the classification method that should
        be used in the model.

    Returns
    -------

    model : SMSGuruModel with `classifier` as classification method

    param_grid : Associated parameter grid
    """

    if classifier == 'svm-b':
        return lda_svm.MODEL, lda_svm.PARAM_GRID_DIM
    elif classifier == 'svm-a':
        return svm.MODEL, svm.PARAM_GRID
    elif classifier == 'lda':
        return lda.MODEL, lda.PARAM_GRID
    elif classifier == 'knn':
        return knn.MODEL, knn.PARAM_GRID
    elif classifier == 'knn-b':
        return knn_b.MODEL, knn_b.PARAM_GRID
    else:
        raise ValueError("Invalid classifier name")
