import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.DataFrame([
                    ['green', 'M', 10.1, 'class1'],
                    ['red', 'L', 13.5, 'class2'],
                    ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
#size_mapping = {
#                    'XL': 3,
#                    'L': 2,
#                    'M': 1}
#df['size'] = df['size'].map(size_mapping)
#inv_size_mapping = {v: k for k, v in size_mapping.items()}
#
#class_mapping = {label:idx for idx,label in
#                enumerate(np.unique(df['classlabel']))}
#df['classlabel'] = df['classlabel'].map(class_mapping)
#
#inv_class_mapping = {v: k for k, v in class_mapping.items()}
#df['classlabel'] = df['classlabel'].map(inv_class_mapping)


X = df[['color', 'size', 'price']].values

ct = ColumnTransformer([("size", OrdinalEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)


print(X)
