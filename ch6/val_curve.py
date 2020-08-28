import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
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

pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1))

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr,
                X=X_train,
                y=y_train,
                param_name='logisticregression__C',
                param_range=param_range,
                cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean,
            color='blue', marker='o',
            markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                    train_mean - train_std, alpha=0.15,
                    color='blue')
plt.plot(param_range, test_mean,
            color='green', linestyle='--',
            marker='s', markersize=5,
            label='validation accuracy')
plt.fill_between(param_range,
                    test_mean + test_std,
                    test_mean - test_std,
                    alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.03])
plt.show()


