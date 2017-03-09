import numpy as np
from sklearn.datasets import make_classification
from stlearn import MultiTaskEstimator
from sklearn.linear_model import (MultiTaskLassoCV,
                                  MultiTaskElasticNet,
                                  MultiTaskElasticNetCV,
                                  ElasticNetCV,
                                  LassoCV, LogisticRegression)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

X, y = make_classification(n_samples=500, random_state=42)
n = 10
Y = np.array(n*[y]).T

mt = MultiTaskEstimator(
    estimator=MultiTaskElasticNetCV(alphas=np.logspace(-3, 3, 7)),
    output_types=n//2*['binary']+n//2*['continuous'])

ls = LogisticRegression()

ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

for train, test in ss.split(X):
    mt.fit(X[train], Y[train])
    ls.fit(X[train], y[train])
    print(ls.score(X[test], y[test]))
    print(mt.score(X[test], Y[test]))
