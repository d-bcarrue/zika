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

ssplit = ShuffleSplit(n_splits=50, test_size=0.25)
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
