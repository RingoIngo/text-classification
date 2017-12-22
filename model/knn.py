import pdb
import collections
import math
import numpy as np

from sklearn.utils import check_array
from sklearn.neighbors import KNeighborsClassifier


def cosine_similarity(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def cosine_semi_metric(x, y):
    return 1 - np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def cosine_dist_to_sim(dist):
    return (dist - 1) * (- 1)


def get_top_n(k, N_class, max_N_class):
    return math.ceil((k * N_class) / max_N_class)


class KNeighborsClassifierB(KNeighborsClassifier):
    def __init__(self, n_jobs=1, n_neighbors=5, **kargs):
        super().__init__(n_jobs=n_jobs, n_neighbors=n_neighbors,
                         metric=cosine_semi_metric, **kargs)
        self.top_n = None

    def fit(self, X, y):
        # compute top_n for each class (label)
        label_distribution = dict(collections.Counter(y))
        max_N_class = max(label_distribution.values())
        self.top_n = {label: get_top_n(self.n_neighbors, N_class, max_N_class)
                      for label, N_class in label_distribution.items()}
        # # list of
        # self.top_n = [get_top_n(self.n_neighbors, label_distribution[label]
        #                         , max_n_class)
        #               for label in self.classes_]
        super().fit(X, y)

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
         ----------
        X : array-like, shape (n_query, n_features), \
            or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        # classes_ = self.classes_
        # _y = self._y
        # if not self.outputs_2d_:
        #     # y = [2, 4, 3, 2] --> y = [[2], [4], [3], [2]]
        #     _y = self._y.reshape((-1, 1))
        #     # c = [2, 3, 4] --> c = [np.array([2,3,4])]
        #     classes_ = [self.classes_]

        # n_outputs = len(classes_)
        # n_samples = X.shape[0]
        # weights = _get_weights(neigh_dist, self.weights)

        # y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        # # why this construct is not quite clear
        # # k=0, classes_k=[2,3,4]
        # for k, classes_k in enumerate(classes_):
        #     # _y[neig_ind, k]=classes of neigbored indices
        #     mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

        #     mode = np.asarray(mode.ravel(), dtype=np.intp)
        #     y_pred[:, k] = classes_k.take(mode)

        # if not self.outputs_2d_:
        #     y_pred = y_pred.ravel()

        # bcz of numerical reasons we don't use the mode approach here

        neigh_sim = cosine_dist_to_sim(neigh_dist)
        # negate sims to sort descending
        neigh_sim_sorted_ind = (-neigh_sim).argsort(axis=1)
        neigh_labels = self._y[neigh_ind]
        # dict where lables are keys and
        # items arrays ([n_samples]), the label probs
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        label_counts = np.empty((n_classes, n_samples))
        for i, label in enumerate(self.classes_):
            label_top_n = self.top_n[label]
            # label_top_n_sim.shape = [n_sample,label_top_n]
            label_top_n_sim = neigh_sim[np.arange(n_samples)[:, np.newaxis], neigh_sim_sorted_ind[:, 0:label_top_n]]
            # label_top_n_labels = neigh_labels[
            #     neigh_sim_sorted_ind[:, 0:label_top_n]]
            label_top_n_labels = neigh_labels[
                np.arange(n_samples)[:, np.newaxis], neigh_sim_sorted_ind[:, 0:label_top_n]]
            # total.shape = [n_samples,]
            total = np.sum(label_top_n_sim, axis=1)
            # pdb.set_trace()
            weighted_counts = np.sum(
                label_top_n_sim * label_top_n_labels == label, axis=1)
            label_counts[i, :] = weighted_counts / total
        y_pred = self.classes_[np.argmax(label_counts, axis=0)]
        return y_pred
