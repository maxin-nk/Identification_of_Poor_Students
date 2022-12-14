import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("poor students identification_datasets.csv", sep=',', header=0, index_col=0)
print(df.head())

start_time = time.time()
data = df.values

frequent_itemsets = apriori(data.drop('id'), min_support=0.1, use_colnames=True)
print(frequent_itemsets)

features = []
for i in range(frequent_itemsets.shape[0]):
    if 'label' in frequent_itemsets.iloc[i, 1]:
        features.append(frequent_itemsets.iloc[i, 1])

df2 = df[features]
X = df2.iloc[:, :-1].values
y = df2.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

k = 4
model = KNeighborsClassifier(n_neighbors=k, weights='distance', p=2, n_jobs=-1, leaf_size=30, metric='minkowski')
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
# print('f1-score train: %.3f, test: %.3f' % (f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)))
print('time: %.3f' % (time.time()-start_time))