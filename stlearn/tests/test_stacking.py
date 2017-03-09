from nose.tools import assert_equal
from nose.tools import assert_true
from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from stlearn import StackingClassifier

n_samples = 200
n_estimators = 2
X0, y = make_classification(n_samples=200, random_state=42)
X1 = X0 ** 2
X = np.c_[X0, X1]

ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)


def test_stacking_essentials():
    """Test initializaing and essential basic function"""
    stacking = StackingClassifier(
        estimators=n_estimators * [LogisticRegression()],
        stacking_estimator=LogisticRegression())
    # assert_equal(getattr(stacking, 'predictions_', None), None)
    assert_equal(stacking.stacking_estimator.__class__,
                 LogisticRegression)
    assert_equal([ee.__class__ for ee in stacking.estimators],
                 n_estimators * [LogisticRegression])
    assert_raises(ValueError, stacking.fit, X[0], y)
    assert_raises(ValueError, stacking.fit, X[:1], y)
    assert_raises(ValueError, stacking.fit, X[:, :1], y)

    stacking.fit(X, y)

    predictions = stacking.predict(X)
    assert_array_equal(np.unique(predictions), np.array([0, 1]))

    score = stacking.score(X, y)
    assert_true(np.isscalar(score))

    predictions_estimators = stacking.predict_estimators(X)
    assert_array_equal(
        predictions_estimators.shape, (n_samples, n_estimators))
    scores_estimators = stacking.score_estimators(X, y)
    assert_equal(len(scores_estimators), n_estimators)
