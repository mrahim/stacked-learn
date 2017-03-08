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


class StackingClassifer(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Meta-classifier of 3D X matrix with labels
    """

    def __init__(self, estimators, stacking_estimator,
                 n_iter=100, memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, random_state=42):
        """ initialization
        """
        self.n_iter = n_iter
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimators = estimators
        self.stacking_estimator = stacking_estimator

    def fit(self, X, y):
        """ Two level training
        X is 3D matrix
        """

        self.estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_estimator)(clf, X, y)
            for _, clf in self.estimators)

        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_estimator)(clf, X)
            for _, clf in self.estimators)

        self.stacking_estimator.fit(predictions_, y)
        return self

    def predict(self, X):
        """ Second level prediction
        X is 3D matrix
        """

        predictions_ = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_estimator)(clf, X)
            for _, clf in self.estimators)

        return self.stacking_estimator.predict(predictions_)

    def score(self, X, y):
        """ Second level accuracy
        """
        return accuracy_score(y, self.predict(X))
