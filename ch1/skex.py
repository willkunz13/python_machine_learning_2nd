from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pdb

X, y = load_iris(return_X_y=True)
pdb.set_trace()
clf = LogisticRegression(random_state=1, max_iter=150).fit(X, y)
clf.predict(X[:2, :])
