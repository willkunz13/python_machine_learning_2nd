from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

tree = DecisionTreeClassifier(criterion='gini',
                                max_depth=4,
                                random_state=1)
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
dot_data = export_graphviz(tree,
                            filled=True,
                            rounded=True,
                            class_names=['Setosa',
                                        'Versicolor',
                                        'Virginica'],
                            feature_names=['petal length',
                                            'petal width'],
                            out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')
