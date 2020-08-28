from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pdb

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                    'machine-learning-databases'
                    '/breast-cancer-wisconsin/wdbc.data',
                    header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                test_size=0.20,
                stratify=y,
                random_state=1)

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1,
                1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range,
                'svc__kernel': ['linear']},
                {'svc__C': param_range,
                'svc__gamma': param_range,
                'svc__kernel': ['rbf']}]

#gs = GridSearchCV(estimator=pipe_svc,
#                    param_grid=param_grid,
#                    scoring='accuracy',
#                    cv=10,
#                    n_jobs=-1)
#gs = gs.fit(X_train, y_train)
#clf = gs.best_estimator_
#clf.fit(X_train, y_train)
#print('Test accuracy: %.3f' % clf.score(X_test, y_test))

gs = GridSearchCV(estimator=pipe_svc,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=2)
scores = cross_val_score(gs, X_train, y_train,
                            scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                        np.std(scores)))

gs = GridSearchCV(estimator=DecisionTreeClassifier(
                    random_state=0),
                param_grid=[{'max_depth': [1, 2, 3,
                                4, 5, 6, 7, None]}],
                scoring='accuracy',
                cv=2)
scores = cross_val_score(gs, X_train, y_train,
                            scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                        np.std(scores)))

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
            s=confmat[i, j],
            va='center', ha='center')
#plt.xlabel('predicted label')
#plt.ylabel('true label')
#plt.show()

print('Precision: %.3f' % precision_score(
            y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(
            y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(
            y_true=y_test, y_pred=y_pred))

scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc,
                    param_grid=param_grid,
                    scoring=scorer,
                    cv=10)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
