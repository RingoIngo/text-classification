"""
The :mod: `classifier_grid_factory`
"""
# Author: Ingo Gühring

import evaluation.lda_svm as lda_svm
# import evaluation.svm
# import evauation.knn


def make_classifier_grid(classifier):
    if classifier == 'lda_svm':
        return lda_svm.make_grid()
#    elif classifier == 'svm':
#        return svm.make_model()
#    elif classifier == 'knn':
#        return knn.make_model()
    else:
        raise ValueError("Invalid classifier name")
