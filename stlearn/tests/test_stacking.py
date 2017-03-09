from nose.tools import assert_equal
from nose.tools import assert_true
from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from stlearn import StackingClassifier
from stlearn import stack_features
from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier


n_samples = 200
n_estimators = 3
X0, y = make_classification(n_samples=200, random_state=42)
# let's say we taks some columns and make them non-linear
X1 = X0[:, :10] ** 2
X2 = X0[:, 10:15] ** 2

X = [X0, X1, X2]
X_stacked, feature_indices = stack_features(X)


def test_stack_features():
    """Test stacking features"""
    X0 = np.array([[1, 2], [3, 4]])
    X1 = np.array([[1, 2, 4], [3, 4, 5]])
    X = [X0, X1]
    X_stacked, features_indices = stack_features(X)
    assert_equal(np.size(X_stacked),
                 np.size(X0) + np.size(X1))
    assert_equal(len(features_indices), len(X))
    assert_equal(X_stacked.shape, (2, 5))


def test_stacking_essentials():
    """Test initializaing and essential basic function"""

    # check inputs
    stacking = assert_raises(
        ValueError, StackingClassifier,
        estimators=2 * [LogisticRegression()],
        feature_indices=feature_indices,
        stacking_estimator=LogisticRegression())

    stacking = assert_raises(
        ValueError, StackingClassifier,
        estimators=n_estimators * [LogisticRegression()],
        feature_indices=feature_indices[:2],
        stacking_estimator=LogisticRegression())

    # test stacking classifier
    stacking = StackingClassifier(
        estimators=[LogisticRegression() for _ in range(3)],
        feature_indices=feature_indices,
        stacking_estimator=LogisticRegression())

    assert_equal(stacking.stacking_estimator.__class__,
                 LogisticRegression)
    assert_equal([ee.__class__ for ee in stacking.estimators],
                 n_estimators * [LogisticRegression])

    stacking.fit(X_stacked, y)

    predictions = stacking.predict(X_stacked)
    assert_array_equal(np.unique(predictions), np.array([0, 1]))

    score = stacking.score(X_stacked, y)
    assert_true(np.isscalar(score))

    predictions_estimators = stacking.predict_estimators(X_stacked)
    assert_array_equal(
        predictions_estimators.shape, (n_samples, n_estimators))
    scores_estimators = stacking.score_estimators(X_stacked, y)
    assert_equal(len(scores_estimators), n_estimators)

    assert_raises(ValueError, stacking.fit, X, y)
    stacking = StackingClassifier(
        estimators=[LogisticRegression() for _ in range(3)],
        feature_indices=[np.array([-500]), np.array([1]), np.array([2])],
        stacking_estimator=LogisticRegression())

    assert_raises(ValueError, stacking.fit, X_stacked, y)

    stacking = StackingClassifier(
        estimators=[LogisticRegression() for _ in range(3)],
        feature_indices=[slice(5000, -5000), slice(1, 10), slice(20)],
        stacking_estimator=LogisticRegression())
    assert_raises(ValueError, stacking.fit, X_stacked, y)


def test_sklearn_high_level():
    stacking = StackingClassifier(
        estimators=[LogisticRegression() for _ in range(3)],
        feature_indices=feature_indices,
        stacking_estimator=LogisticRegression())
    assert_true(is_classifier(stacking))
    scores = cross_val_score(X=X_stacked, y=y, estimator=stacking)
    assert_equal(len(scores), 3)
