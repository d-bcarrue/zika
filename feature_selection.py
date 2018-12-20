#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle

# feature selection
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
ydata = y.values

ssplit = ShuffleSplit(n_splits=100, test_size=0.2)
importances = np.zeros(Xdata.shape[1])
for i_train, i_test in ssplit.split(Xdata):
    X_train, y_train = Xdata[i_train, :], ydata[i_train]
    alg = RandomForestClassifier(n_estimators=100)
    alg.fit(X_train, y_train)
    importances = np.row_stack((importances, alg.feature_importances_))

importances = importances[1:, :]
importances_mean = np.mean(importances, axis=0)
importances_std = np.std(importances, axis=0)

indices = np.argsort(importances_mean)[::-1]

plt.figure()
plt.title('Feature importances')
plt.bar(range(X.shape[1]), importances_mean[indices], color='b',
        yerr=importances_std[indices])
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
indices = indices[::-1]
scores_array_val = np.zeros(100)
scores_array_test = np.zeros(100)
progreso = 0
for i in range(len(indices)):
    scores_val = []
    scores_test = []
    Xnuevo = Xdata[:, indices[i:]]
    ssplit = ShuffleSplit(n_splits=100, test_size=0.2)

    for i_train, i_test in ssplit.split(Xnuevo):
        X_train, X_test = Xnuevo[i_train, :], Xnuevo[i_test, :]
        y_train, y_test = ydata[i_train], ydata[i_test]
        alg = RandomForestClassifier(n_estimators=100)
        alg.fit(X_train, y_train)
        scores_val.append(alg.score(X_train, y_train))
        scores_test.append(alg.score(X_test, y_test))

    scores_array_val = np.row_stack((scores_array_val, scores_val))
    scores_array_test = np.row_stack((scores_array_test, scores_test))
    progreso += 100/47
    print('{}%'.format(round(progreso)))

scores_array_val = scores_array_val[1:, :]
scores_array_test = scores_array_test[1:, :]

scores_val_mean = np.mean(scores_array_val, axis=1)
scores_test_mean = np.mean(scores_array_test, axis=1)

scores_val_std = np.std(scores_array_val, axis=1)
scores_test_std = np.std(scores_array_test, axis=1)

n = list(range(len(indices), 0, -1))
plt.figure()
plt.xlabel('n_features')
plt.ylabel('accuracy')
plt.grid()
plt.fill_between(n, scores_val_mean + scores_val_std,
                 scores_val_mean - scores_val_std, color='r', alpha=0.1)
plt.fill_between(n, scores_test_mean + scores_test_std,
                 scores_test_mean - scores_test_std, color='b', alpha=0.1)
plt.plot(n, scores_val_mean, 'ro-', label='Training score')
plt.plot(n, scores_test_mean, 'bo-', label='Test score')
plt.legend(loc='best')
plt.show()
