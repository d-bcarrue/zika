#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     ParameterGrid)
from sklearn.neural_network import MLPClassifier

df = pd.read_csv(os.path.join('ds_Zika.csv'))
nposit = len(df[df['Output'] == 1])
inegat = df[df.Output == 0].index

random_indices = np.random.choice(inegat, nposit, replace=False)
iposit = df[df.Output == 1].index
new_indices = np.concatenate([iposit, random_indices])
df = df.loc[new_indices]
df = shuffle(df).reset_index(drop=True)

X = df.drop('Output', axis=1)
y = df['Output']

Xdata = X.values
Ydata = y.values

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,
                                                    test_size=0.25,
                                                    stratify=Ydata)


#alg = RandomForestClassifier()
alg = MLPClassifier()
param_grid = {'solver': ['lbfgs'], 'max_iter': [500,1000,1500],
              'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12)}
              #'random_state':[0,1,2,3,4,5,6,7,8,9]}
grid = GridSearchCV(alg, param_grid=param_grid, cv=10, n_jobs=-1)

# el numero de estimadores(numero de arboles en el bosque), es demasiado bajo
# quiza probar mayor numero
# param_grid = {'n_estimators': range(1, 51, 5),
#               'max_depth': range(1, 25, 2),
#               'min_samples_leaf': range(1, 10, 2)}

#grid = GridSearchCV(alg, param_grid=param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print('validation accuracy:', grid.score(X_train, y_train))
print('test accuracy:', grid.score(X_test, y_test))

#%%
c = pd.DataFrame(grid.cv_results_)
c['train_test_difference'] = c.mean_train_score - c.mean_test_score
#print(c[(c.mean_test_score >= 0.90) & (c.train_test_difference <= 0.02)][(['mean_train_score', 'mean_test_score','train_test_difference'])].sort_values('train_test_difference'))
#i = 197

c_sort = c.sort_values(by=['train_test_difference','mean_test_score'])
i = best.first_valid_index()


#max_depth = c.param_max_depth.iloc[i]
#min_samples_leaf = c.param_min_samples_leaf.iloc[i]
#n_estimators = c.param_n_estimators.iloc[i]
#%%
#alg1 = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)

#alg1.fit(X_train, y_train)

#print('validation accuracy:', alg1.score(X_train, y_train))
#print('test accuracy:', alg1.score(X_test, y_test))
#y_predicted_val = alg1.predict(X_train)
#y_predicted_test = alg1.predict(X_test)







