# -*- coding: utf-8 -*-
"""
Prediction stacking API
"""
# Author: Mehdi Rahim <rahim.mehdi@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.externals.joblib import Memory, Parallel, delayed


def stack_features(X):
    """Stack features from sources

    Parameters
    ----------
    X : a list of 2d matrices

    Returns
    -------
    Xstacked : (n_samples x (n_features*n_sources)) stacked 2d matrix

    features_indices : (n_features*n_sources) list of indices
    """
    X_stacked = np.hstack(X)

    features_markers = np.r_[0, np.cumsum([x.shape[1] for x in X])]
    feature_indices = [slice(features_markers[i],
                             features_markers[i+1])
                       for i in range(len(features_markers)-1)]

    return X_stacked, feature_indices


def _split_features(X, feature_indices):
    """helper"""
    return [X[:, fi] for fi in feature_indices]


def _fit_estimator(clf, X, y):
    """Helper to fit estimator"""
    return clf.fit(X, y)


def _predict_estimator(clf, X):
    """Helper tor predict"""
    return clf.predict(X)


def _predict_proba_estimator(clf, X):
    """Helper to get prediction method"""
    # try predict_proba
    predict_proba = getattr(clf, "predict_proba", None)
    if callable(predict_proba):
        return clf.predict_proba(X)[:, 0]

    # or decision_function
    decision_function = getattr(clf, "decision_function", None)
    if callable(decision_function):
        return clf.decision_function(X)

    raise NotImplementedError("predict_proba not supported")


def _check_Xy(stacking, X, y=None):
    """check dimensions"""
    if np.ndim(X) != 3:
        raise ValueError(
            'X must be 3 dimensional, your X has %d dimensions' % np.ndim(X))
    expected_n_sources = len(stacking.estimators)
    if expected_n_sources != np.asarray(X).shape[0]:
        raise ValueError(
            'The first axis of X (%d) should match the '
            'number of estimators (%d)' % (
                X.shape[0],
                len(stacking.estimators)))
    if y is not None:
        if len(y) != np.asarray(X).shape[1]:
            raise ValueError(
                'The second axis of X (%d) should match the '
                'number of samples (%d)' % (
                    X.shape[1],
                    len(stacking.estimators)))


class StackingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Stacking Meta-classifier of 3D X matrix with labels

    Parameters
    ----------
    estimators : list of Estimator objects compatible with scikit-learn
        The estimators to be used with each source of inputs. Length must match
        the firt dimensions of X.
    stacking_estimator : Estimator objects compatible with scikit-learn
        The estimator used to integrate the predictions of the estimators.
    memory : joblib memory object | None
        The caching configuration. Defaults to `Memory(cachedir=None)`.
    memory_level : int (defaults to 0)
        The memory level used for caching.
    """

    def __init__(self, estimators=None,
                 stacking_estimator=None,
                 feature_indices=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1):

        self.estimators = estimators
        self.stacking_estimator = stacking_estimator
        self.feature_indices = feature_indices
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs

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

        X_list = _split_features(X, self.feature_indices)
        _check_Xy(self, X_list, y)
        self.estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(clf, x, y)
            for x, clf in zip(X_list, self.estimators))

        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_estimator)(clf, x)
            for x, clf in zip(X_list, self.estimators))
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
        X_list = _split_features(X, self.feature_indices)
        _check_Xy(self, X_list)
        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_estimator)(clf, x)
            for x, clf in zip(X_list, self.estimators))
        predictions_ = np.array(predictions_).T

        return self.stacking_estimator.predict(predictions_)

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
        X_list = _split_features(X, self.feature_indices)
        _check_Xy(self, X_list)
        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_estimator)(clf, x)
            for x, clf in zip(X_list, self.estimators))
        return np.array(predictions_).T

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
