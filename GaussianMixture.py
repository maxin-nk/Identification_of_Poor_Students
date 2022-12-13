import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import time


df = pd.read_csv("poor students identification_datasets.csv", sep=',', header=0, index_col=0)
print(df.head())

start_time = time.time()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

k = 5
model = GaussianMixture(n_components=k, max_iter=200, tol=1e-04)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
print('f1-score train: %.3f, test: %.3f' % (f1_score(y_train, y_train_pred, average='micro'), f1_score(y_test, y_test_pred, average='micro')))
print('time: %.3f' % (time.time()-start_time))