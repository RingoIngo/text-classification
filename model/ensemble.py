"""
The :mod:
"""
# Author: Ingo Guehring

from time import gmtime, strftime
import numpy as np
import warnings
from functools import partial
from itertools import product
from collections import defaultdict
from functools import reduce
import os

from scipy.stats import rankdata

from sklearn.ensemble import VotingClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._search import BaseSearchCV, _check_param_grid
from sklearn.base import is_classifier, clone
from sklearn.utils.validation import indexable
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score,\
       _aggregate_score_dicts
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.deprecation import DeprecationDict
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.metrics import f1_score


def f1_macroB(estimator, X, y):
    """Wrapper function for f1 score that passes y down

    This scorer can only be used with an estimator that
    implements the 'set_y_true()' function. So far
    this is GridSearchCVB, where the underlying estimator
    is VotingClassifierB. y_true is passed down so that
    it can be saved together with the calculated probs.
    """
    estimator.set_y_true(y)
    y_pred = estimator.predict(X)
    return f1_score(y, y_pred, average='macro')


class VotingClassifierB(VotingClassifier):
    """
    Parameters
    -------------
    comb_method : str, {'avg', 'mult'} (default='avg')
        If 'avg' (weighted) averaging is used to combine the output
        of the classifiers. If 'mult' combination is done with the
        multiplying rule in 'Combining multiple classi"ers by averaging or by
        multiplying?' by M.Tax et al.

    """
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=1,
                 flatten_transform=None, save_avg=False,
                 save_avg_path='./results/gridsearch/ensemble/raw/',
                 comb_method='avg'):
        super().__init__(estimators, voting=voting, weights=weights,
                         n_jobs=n_jobs,
                         flatten_transform=flatten_transform)
        self.save_avg = save_avg
        self.save_avg_path = save_avg_path
        self.comb_method = comb_method
        self.y_true = None

    def set_y_true(self, y_true):
        """Set the true labels for the samples that have to be
        predicted. This is done so that the true labels can be saved
        together with the computed probs.
        """
        self.y_true = y_true

    def fit(self, X, y, sample_weight=None):
        """Estimate the class prior probabilities

        When the computed probabilities from the differenct
        classifiers are combined via multiplication the class
        priors are needed.
        """
        super().fit(X, y, sample_weight=sample_weight)
        if self.comb_method == 'mult':
            transform_y = self.le_.transform(y)
            classes = np.arange(len(self.classes_))
            self.priors = np.asarray(
                [np.sum(transform_y == i)/len(transform_y) for i in classes])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting and save probs

        If comb_method='avg' then the combined probabilites are computed with
        the original average method. If comb_method='mult' the combined
        probabilities are computed with a multiplication method described in
        a paper by Tax et. al..
        If self.save_avg then the computed combined probabilities and the
        true labels are saved, together with the probabilities of the first
        estimator in self.esttimators_. This should be the estimator where the
        best (single) performance is expected und can be used to analyse the
        gain from the ensemble method.
        Note that it makes no sense to compute both combination methods at
        the same time, since in the previous gridsearch the parameters where
        optimized with respect to one method only. So get the probabilities
        for both methods the whole nested cross-validation has to be run
        two times.
        """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators_')
        collected_probas = self._collect_probas(X)

        # combine probabilities with averaging
        if self.comb_method == 'avg':
            combined_probs = np.average(collected_probas, axis=0,
                                        weights=self._weights_not_none)

        # combine probabilities with multiplication method
        if self.comb_method == 'mult':
            R = len(self.estimators)
            combined_probs = reduce(lambda X, Y: X*Y,
                                    collected_probas)*(self.priors**(R - 1))
            # normalize
            combined_probs = (
                combined_probs.T/(np.sum(combined_probs, axis=1)).T).T

        # save combined probabilities, probabilities of first (best) single
        # estimator, true class labels and labels.
        if self.save_avg:
            current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            filename = self.save_avg_path + current_time + 'avg'
            savename = filename
            suffix = 1
            while os.path.exists(savename + '.npz'):
                savename = filename + '-' + str(suffix)
                suffix = suffix + 1
            print('save avg to file: ' + savename)
            np.savez(
                savename, probs=combined_probs,
                best_single=collected_probas[0, :, :], y_true=self.y_true,
                classes=self.le_.classes_)
        return combined_probs


class BaseSearchCVB(BaseSearchCV):
    """Base class for hyper parameter search with cross-validation."""

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        # Regenerate parameter iterable for each fit
        candidate_params = list(self._get_param_iterator())
        n_candidates = len(candidate_params)
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(delayed(_fit_and_score)(clone(base_estimator), X, y, scorers, train,
                                  test, self.verbose, parameters,
                                  fit_params=fit_params,
                                  return_train_score=self.return_train_score,
                                  return_n_test_samples=True,
                                  return_times=True, return_parameters=False,
                                  error_score=self.error_score)
          for parameters, (train, test) in product(candidate_params,
                                                   cv.split(X, y, groups)))

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
             score_time) = zip(*out)
        else:
            (test_score_dicts, test_sample_counts, fit_time,
             score_time) = zip(*out)

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)
        if self.return_train_score:
            train_scores = _aggregate_score_dicts(train_score_dicts)

        # TODO: replace by a dict in 0.21
        results = (DeprecationDict() if self.return_train_score == 'warn'
                   else {})

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)
        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if self.iid else None)
            if self.return_train_score:
                prev_keys = set(results.keys())
                _store('train_%s' % scorer_name, train_scores[scorer_name],
                       splits=True)

                if self.return_train_score == 'warn':
                    for key in set(results.keys()) - prev_keys:
                        message = (
                            'You are accessing a training score ({!r}), '
                            'which will not be available by default '
                            'any more in 0.21. If you need training scores, '
                            'please set return_train_score=True').format(key)
                        # warn on key access
                        results.add_warning(key, message, FutureWarning)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = candidate_params[self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_]

        if self.refit:
            self.best_params_['save_avg'] = True
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class GridSearchCVB(BaseSearchCVB):
    """
    - set save_avg in refitted best estimator
    - pass y_true down to the underlying estmator
    """
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score="warn"):
        super(GridSearchCVB, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def set_y_true(self, y_true):
        """Set the true class labels of the samples that have to be predicted

        This works only if the underlying estimator supports 'set_y_true()'.
        So far this only for VotingClassifierB the case.
        """
        self.best_estimator_.set_y_true(y_true)

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""
        return ParameterGrid(self.param_grid)
