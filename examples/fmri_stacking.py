from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from stlearn import StackingClassifier

n = 20
X, y = make_classification(n_samples=200, random_state=42)
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

stacking = StackingClassifier(estimators=n*[LogisticRegression()],
                              stacking_estimator=LogisticRegression())

for train, test in ss.split(X):
    stacking.fit(n*[X[train]], y[train])
    print(stacking.score(n*[X[test]], y[test]))
    print(stacking.score_estimators(n*[X[test]], y[test]))
