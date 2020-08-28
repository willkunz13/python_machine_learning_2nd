from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from plot_decision_regions import *
#import logisticregressiongd as lrd
import pdb

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#from sklearn.linear_model import Perceptron

#ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
#ppn.fit(X_train_std, y_train)
#y_pred = ppn.predict(X_test_std)
#print('Misclassified samples: %d' % (y_test != y_pred).sum())

#from sklearn.metrics import accuracy_score

#print('Accuracy: &.2f' % accuracy_score(y_test, y_pred)

#X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
#y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
#lrgd = lrd.LogisticRegressionGD(eta=0.05,
#    n_iter=1000,
#    random_state=1)
#lrgd.fit(X_train_01_subset,
#    y_train_01_subset)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                        y_combined,
                        classifier=lr,
                        test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
pdb.set_trace()
print( lr.predict_proba(X_test_std[:3, :1]).argmax(axis=1))
