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
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import (
    MultiTaskLassoCV, MultiTaskElasticNetCV, LogisticRegression)


class MultiTaskEstimator(BaseEstimator, TransformerMixin):
    """MultiTask estimator for multiple (continuous / discrete) outputs.

    Parameters
    ----------
    estimator : Multitask scikit-learn estimator, can be
                {"MultiTaskLasso", "MultiTaskLassoCV",
                 "MultiTaskElasticNet", "MultiTaskElasticNetCV"}

    output_types : shape = (n_outputs,) type of each output, can be
                    {"binary", "continuous"}
    """

    def __init__(self, estimator=None, output_types=None):
        self.estimator = estimator
        # check if output types are okay
        for output in output_types:
            if output not in ['binary', 'continuous']:
                raise TypeError('Unrecognized output type: %s' % output)
        self.output_types = output_types
        self.n_outputs = len(self.output_types)

    def fit(self, X, Y):
        """Fit estimator to the given training data and all outputs.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        Y : array-like, shape (n_samples, n_outputs)
            Target matrix relative to X.
        """

        if Y.shape[1] != self.n_outputs:
            raise ValueError('Y columns=%u whereas n_outputs=%u'
                             % (Y.shape[1], self.output_types))
        # model fitting
        self.estimator.fit(X, Y)
        return self

    def _decision_function(self, X):
        return self.estimator._decision_function(X)

    def predict(self, X):
        """Predict outputs for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Sample matrix.

        Returns
        -------
        Y_pred : array, shape = (n_samples, n_outputs)
            Predicted outputs per sample.
        """
        # predict multiple outputs
        Y_pred = self._decision_function(X)
        for i in range(self.n_outputs):
            if self.output_types[i] == 'binary':
                # binarize classification results
                labels = np.zeros(Y_pred[:, i].shape)
                labels[Y_pred[:, i] >= 0.5] = 1
                Y_pred[:, i] = labels
        return Y_pred

    def score(self, X, Y):
        """Returns accuracy for each outputs.

        r2_score is used for continuous outputs,
        accuracy_score is used for binary outputs.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Sample matrix.

        Y : array-like, shape = (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : list of float, shape (n_outputs,)
            Accuracy of self.predict(X) wrt. Y.
        """
        Y_pred = self.predict(X)
        scores = np.empty((self.n_outputs))
        for i in range(self.n_outputs):
            # accuracy_score for classification
            if self.output_types[i] == 'binary':
                scores[i] = accuracy_score(Y[:, i], Y_pred[:, i])
            # r2_score for regression
            elif self.output_types[i] == 'continuous':
                scores[i] = r2_score(Y[:, i], Y_pred[:, i])
        return scores
