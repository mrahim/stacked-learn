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


def fit_estimator(clf, X, y):
    return clf.fit(X, y)


def predict_estimator(clf, X):
    return clf.predict(X)


def predict_proba_estimator(clf, X):
    # try predict_proba
    predict_proba = getattr(clf, "predict_proba", None)
    if callable(predict_proba):
        return clf.predict_proba(X)[:, 0]

    # or decision_function
    decision_function = getattr(clf, "decision_function", None)
    if callable(decision_function):
        return clf.decision_function(X)

    raise NotImplementedError("predict_proba not supported")


class StackingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Meta-classifier of 3D X matrix with labels
    """

    def __init__(self, estimators=None,
                 stacking_estimator=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1):
        """ initialization
        """
        self.estimators = estimators
        self.stacking_estimator = stacking_estimator
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """ stacking model fitting
        X is 3D matrix
        """

        self.estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_estimator)(clf, x, y)
            for x, clf in zip(X, self.estimators))

        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_proba_estimator)(clf, x)
            for x, clf in zip(X, self.estimators))
        predictions_ = np.array(predictions_).T

        self.stacking_estimator.fit(predictions_, y)
        return self

    def predict(self, X):
        """ stacking model prediction
        X is 3D matrix
        """

        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_proba_estimator)(clf, x)
            for x, clf in zip(X, self.estimators))
        predictions_ = np.array(predictions_).T

        return self.stacking_estimator.predict(predictions_)

    def score(self, X, y):
        """ stacking model accuracy
        """
        return accuracy_score(y, self.predict(X))

    def predict_estimators(self, X):
        """ prediction from separate estimators
        """
        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_estimator)(clf, x)
            for x, clf in zip(X, self.estimators))
        return np.array(predictions_).T

    def score_estimators(self, X, y):
        """ accuracy from separate estimators
        """
        predictions_ = self.predict_estimators(X)
        return np.array([accuracy_score(y, p) for p in predictions_.T])
