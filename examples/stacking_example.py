import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from stlearn import StackingClassifier, stack_features

n = 20
X, y = make_classification(n_samples=200, random_state=42)
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

X_stacked, features_indices = stack_features(n*[X])

stacking = StackingClassifier(estimators=n*[LogisticRegression()],
                              stacking_estimator=LogisticRegression(),
                              feature_indices=features_indices)

for train, test in ss.split(X_stacked):
    stacking.fit(X_stacked[train], y[train])
    print(stacking.score(X_stacked[test], y[test]))
    print(stacking.score_estimators(X_stacked[test], y[test]))
