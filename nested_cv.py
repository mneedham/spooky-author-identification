from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np

params = [{'C': [0.01, 0.05, 0.1, 1]}, {'n_estimators': [10, 100, 1000]}]
models = [SVR(), RandomForestRegressor()]

df = load_boston()
X = df['data']
y = df['target']

cv = [[] for _ in range(len(models))]
for tr, ts in KFold(len(X)):
    for i, (model, param) in enumerate(zip(models, params)):
        best_m = GridSearchCV(model, param)
        best_m.fit(X[tr], y[tr])
        s = mean_squared_error(y[ts], best_m.predict(X[ts]))
        print(model, param, s)
        cv[i].append(s)
print(np.mean(cv, 1))