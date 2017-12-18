"""
The :mod: `classifier_grid_factory`
"""
# Author: Ingo Guehring

import evaluation.lda_svm as lda_svm
import evaluation.svm as svm
# import evauation.knn


def make_classifier_grid(classifier):
    if classifier == 'lda_svm':
        return lda_svm.MODEL, lda_svm.PARAM_GRID
    elif classifier == 'svm':
        return svm.MODEL, svm.PARAM_GRID
#    elif classifier == 'knn':
#        return knn.make_model()
    else:
        raise ValueError("Invalid classifier name")
