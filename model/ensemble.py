"""
The :mod:
"""
# Author: Ingo Guehring

from time import gmtime, strftime
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.utils.validation import check_is_fitted


class VotingClassifierB(VotingClassifier):
    """
    """
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=1,
                 flatten_transform=None, save_avg=False):
        super().__init__(estimators, voting='hard', weights=None, n_jobs=1,
                         flatten_transform=None)
        self.save_avg = save_avg

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators_')
        avg = np.average(self._collect_probas(X), axis=0,
                         weights=self._weights_not_none)
        if self.save_avg:
            current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            path = './results/gridsearch/ensemble/raw/'
            filename = current_time + 'avg'
            np.savez(path + filename, avg=avg, classes=self.le_.classes_)
            return avg
