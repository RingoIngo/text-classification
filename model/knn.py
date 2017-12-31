import collections
import math
import numpy as np

from sklearn.utils import check_array
from sklearn.neighbors import KNeighborsClassifier


def cosine_similarity(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def cosine_semi_metric(x, y):
    # handle zero vectors
    if (not np.any(x) and np.any(y)) or (not np.any(y) and np.any(x)):
        return 1
    if not np.any(x) and not np.any(y):
        return 0
    else:
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

        # bcz of numerical reasons we don't use the mode approach here

        neigh_sim = cosine_dist_to_sim(neigh_dist)
        # self._y does not contain labels!
        # it contains the indices of the labels in self.classes_
        # see SupervisedIntegerMixin fit() method
        neigh_labels = self._y[neigh_ind]
        # dict where lables are keys and
        # items arrays ([n_samples]), the label probs
        n_samples = X.shape[0]
        classes = self.classes_
        n_classes = len(classes)
        label_counts = np.empty((n_classes, n_samples))
        for i, label in enumerate(classes):
            label_top_n = self.top_n[label]
            # label_top_n_sim.shape = [n_sample,label_top_n]
            label_top_n_sim = neigh_sim[
                np.arange(n_samples)[:, np.newaxis], np.arange(label_top_n)]
            label_top_n_labels_idx = neigh_labels[
                np.arange(n_samples)[:, np.newaxis], np.arange(label_top_n)]
            # total.shape = [n_samples,]
            total = np.sum(label_top_n_sim, axis=1)
            label_idx = np.where(classes == label)[0][0]
            weighted_counts = np.sum(
                label_top_n_sim * (label_top_n_labels_idx == label_idx),
                axis=1)
            label_counts[i, :] = weighted_counts / total
        y_pred = self.classes_[np.argmax(label_counts, axis=0)]
        return y_pred

    def predict_proba(self, X):
        """Return probability estimates for the test data X.
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

        # bcz of numerical reasons we don't use the mode approach here

        neigh_sim = cosine_dist_to_sim(neigh_dist)
        # self._y does not contain labels!
        # it contains the indices of the labels in self.classes_
        # see SupervisedIntegerMixin fit() method
        neigh_labels = self._y[neigh_ind]
        # dict where lables are keys and
        # items arrays ([n_samples]), the label probs
        n_samples = X.shape[0]
        classes = self.classes_
        n_classes = len(classes)
        label_counts = np.empty((n_classes, n_samples))
        for i, label in enumerate(classes):
            label_top_n = self.top_n[label]
            # label_top_n_sim.shape = [n_sample,label_top_n]
            label_top_n_sim = neigh_sim[
                np.arange(n_samples)[:, np.newaxis], np.arange(label_top_n)]
            label_top_n_labels_idx = neigh_labels[
                np.arange(n_samples)[:, np.newaxis], np.arange(label_top_n)]
            # total.shape = [n_samples,]
            total = np.sum(label_top_n_sim, axis=1)
            label_idx = np.where(classes == label)[0][0]
            weighted_counts = np.sum(
                label_top_n_sim * (label_top_n_labels_idx == label_idx),
                axis=1)
            label_counts[i, :] = weighted_counts / total
        normalizer = label_counts.sum(axis=0)
        normalizer[normalizer == 0] = 1.0
        label_counts = label_counts / normalizer
        return label_counts.T
