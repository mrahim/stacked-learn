# -*- coding: utf-8 -*-
"""
    Multitask prediction API
"""
# Author: Mehdi Rahim <rahim.mehdi@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.linear_model import (
    MultiTaskLassoCV, MultiTaskElasticNetCV, LogisticRegression)


class MultiTaskEstimator(BaseEstimator, TransformerMixin):
    """MultiTask estimator for multiple (continuous / discrete) outputs.
    """

    def __init__(self, estimator=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, output_types=None):
        self.estimator = estimator
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        # check if output types are okay
        for output in output_types:
            if output not in ['binary', 'continuous']:
                raise TypeError('Unrecognized output type: %s' % output)
        self.output_types = output_types
        self.n_outputs = len(self.output_types)

    def fit(self, X, Y):
        if Y.shape[1] != self.n_outputs:
            raise ValueError('Y columns=%u whereas n_outputs=%u'
                             % (Y.shape[1], self.output_types))
        # model fitting
        self.estimator.fit(X, Y)
        return self

    def _decision_function(self, X):
        return self.estimator._decision_function(X)

    def predict(self, X):
        # predict multiple outputs
        Ypred = self._decision_function(X)
        for i in range(self.n_outputs):
            if self.output_types[i] == 'binary':
                # binarize classification results
                labels = np.zeros(Ypred[:, i].shape)
                labels[Ypred[:, i] >= 0.5] = 1
                Ypred[:, i] = labels
        return Ypred

    def score(self, X, Y):
        """Returns accuracy for each outputs.

        r2_score is used for continuous outputs,
        accuracy_score is used for binary outputs.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The multi-input samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : list of float, shape (n_outputs,)
            Mean accuracy of self.predict(X) wrt. Y.
        """
        # predict multiple outputs
        # accuracy for regression and classification
        Ypred = self.predict(X)
        scores = np.empty((self.n_outputs))
        for i in range(self.n_outputs):
            if self.output_types[i] == 'binary':
                scores[i] = accuracy_score(Y[:, i], Ypred[:, i])
            elif self.output_types[i] == 'continuous':
                scores[i] = r2_score(Y[:, i], Y_pred[:, i])
        return scores
