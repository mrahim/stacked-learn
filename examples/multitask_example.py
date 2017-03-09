import numpy as np
from sklearn.datasets import make_classification
from stlearn import MultiTaskEstimator
from sklearn.linear_model import (MultiTaskLassoCV,
                                  MultiTaskElasticNet,
                                  MultiTaskElasticNetCV,
                                  LassoCV, LogisticRegression)
from sklearn.metrics import accuracy_score

X, y = make_classification(random_state=25)
n = 10

Y = np.array(n*[y]).T

mt = MultiTaskEstimator(
                        # estimator=MultiTaskLassoCV(alphas=np.logspace(-3, 3, 7)),
                        estimator=MultiTaskElasticNetCV(),
                        output_types=n*['binary'])

mt.fit(X, Y)
print(mt.score(X, Y))
