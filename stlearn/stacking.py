# -*- coding: utf-8 -*-
"""
Prediction stacking API
"""
# Author: Mehdi Rahim <rahim.mehdi@gmail.com>
#         Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.externals.joblib import Parallel, delayed


def stack_features(X):
    """Stack features from sources

    Parameters:
    -----------
    X : list of array-like (n_samples, n_features)
        The data to be used as source for each estimator. The first
        dataset corresponds to the first estimator.

    Returns:
    --------
    X_stacked : array, (n_samples, n_features)
        The stacked data, such that the number of features corresponds
        to the sum of number of featrues in each source.

    features_indices : list of indexers
        Index epxressions to be applied on the columns of X_stacked.
        Can be slices, lists of intgers or bool.
    """
    X_stacked = np.hstack(X)

    features_markers = np.r_[0, np.cumsum([x.shape[1] for x in X])]
    feature_indices = [slice(features_markers[i],
                             features_markers[i + 1])
                       for i in range(len(features_markers) - 1)]

    return X_stacked, feature_indices


def _split_features(X, feature_indices):
    """Helper"""
    return [X[:, fi] for fi in feature_indices]


def _fit_estimator(clf, X, y):
    """Helper to fit estimator"""
    return clf.fit(X, y)


def _predict_estimator(clf, X):
    """Helper tor predict"""
    return clf.predict(X)


def _predict_proba_estimator(clf, X):
    """Helper to get prediction method"""

    # XXX this is not safe. Maybe add explicit 1st level scoring param.
    # try predict_proba
    predict_proba = getattr(clf, "predict_proba", None)
    if callable(predict_proba):
        return clf.predict_proba(X)

    # or decision_function
    decision_function = getattr(clf, "decision_function", None)
    if callable(decision_function):
        return clf.decision_function(X)

    raise NotImplementedError("predict_proba not supported")


def _check_Xy(stacking, X, y=None):
    """check dimensions"""
    if np.ndim(X) != 2:
        raise ValueError('X_stacked must be a 2D array')

    for ii, feat_inds in enumerate(stacking.feature_indices):
        if not isinstance(X, np.ndarray):
            raise ValueError('You have something else than an array in X[%d]'
                             % ii)
        if isinstance(feat_inds, (list, tuple, np.ndarray)):
            this_max = np.max(feat_inds)
            this_min = abs(np.min(feat_inds))
            if this_max >= X.shape[1] or this_min > X.shape[1]:
                raise ValueError('On source %s your indexer is out of bound'
                                 % ii)
        elif isinstance(feat_inds, slice):
            stop = feat_inds.stop
            start = feat_inds.start
            if start is None:
                start = 0
            if stop is None:
                stop = -1
            if (start >= X.shape[1] or abs(stop) > X.shape[1]):
                ValueError('Your slices are bad and generate empty views')


class StackingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Stacking Meta-classifier of 3D X matrix with labels

    Parameters
    ----------
    estimators : list of Estimator objects compatible with scikit-learn
        The estimators to be used with each source of inputs. Length must match
        the firt dimensions of X.

    stacking_estimator : Estimator objects compatible with scikit-learn
        The estimator used to integrate the predictions of the estimators.

    features_indices : list of indexers
        Index epxressions to be applied on the columns of X_stacked.
        Can be slices, lists of intgers or bool.

    n_jobs : int (default: 1)
        The number of jobs to run in parallel (across estimators).
    """

    def __init__(self, estimators,
                 stacking_estimator,
                 feature_indices,
                 n_jobs=1):

        if len(estimators) != len(feature_indices):
            raise ValueError('The estimators and feature indices must be of '
                             'the same lenghts')

        if len(set(estimators)) < len(estimators):
            raise ValueError('Estimators must be indpendent')
        self.estimators = estimators
        self.stacking_estimator = stacking_estimator
        self.feature_indices = feature_indices
        self.n_jobs = n_jobs

    def _disambiguate_probability(self, x):
        return x[:, -1] if np.ndim(x) > 1 else x

    def fit(self, X, y):
        """Fit all estimators according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.
        """

        _check_Xy(self, X, y)
        X_list = _split_features(X, self.feature_indices)

        self.estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(clf, x, y)
            for x, clf in zip(X_list, self.estimators))

        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_estimator)(clf, x)
            for x, clf in zip(X_list, self.estimators))
        predictions_ = [self._disambiguate_probability(x)
                        for x in predictions_]
        predictions_ = np.array(predictions_).T

        self.stacking_estimator.fit(predictions_, y)
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            The multi-input samples.

        Returns
        -------
        C : array, shape = (n_samples)
            Predicted class label per sample.
        """
        _check_Xy(self, X)
        X_list = _split_features(X, self.feature_indices)
        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_estimator)(clf, x)
            for x, clf in zip(X_list, self.estimators))
        predictions_ = [self._disambiguate_probability(x)
                        for x in predictions_]
        predictions_ = np.array(predictions_).T

        return self.stacking_estimator.predict(predictions_)

    def predict_proba(self, X):
        """Predict class probability for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            The multi-input samples.

        Returns
        -------
        C : array, shape = (n_samples)
            Predicted class label per sample.
        """
        _check_Xy(self, X)
        X_list = _split_features(X, self.feature_indices)
        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_estimator)(clf, x)
            for x, clf in zip(X_list, self.estimators))
        predictions_ = [self._disambiguate_probability(x)
                        for x in predictions_]
        predictions_ = np.array(predictions_).T

        return _predict_proba_estimator(self.stacking_estimator, predictions_)

    def decision_function(self, X):
        return self._disambiguate_probability(self.predict_proba(X))

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The multi-input samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.


        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return accuracy_score(y, self.predict(X))

    def predict_estimators(self, X):
        """Predict class labels for samples in X for each estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            The multi-input samples.

        Returns
        -------
        C : array, shape = (n_samples, n_estimators)
            Predicted class label per sample and estimators.
        """
        _check_Xy(self, X)
        X_list = _split_features(X, self.feature_indices)
        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_estimator)(clf, x)
            for x, clf in zip(X_list, self.estimators))
        return np.array(predictions_).T

    def predict_proba_estimators(self, X):
        """Predict class label probability for samples in X for each estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            The multi-input samples.

        Returns
        -------
        C : array, shape = (n_samples, n_classes, n_estimators)
            Predicted class label per sample and estimators.
        """
        _check_Xy(self, X)
        X_list = _split_features(X, self.feature_indices)
        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_estimator)(clf, x)
            for x, clf in zip(X_list, self.estimators))
        return np.transpose(np.array(predictions_),
                            (1, 2, 0))

    def score_estimators(self, X, y):
        """Returns the mean accuracy for each estimators.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The multi-input samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : list of float, shape (n_estimators,)
            Mean accuracy of self.predict_estimators(X) wrt. y.
        """
        predictions_ = self.predict_estimators(X)
        return np.array([accuracy_score(y, p) for p in predictions_.T])
