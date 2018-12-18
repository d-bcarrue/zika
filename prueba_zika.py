#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import roc_curve,auc,confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 

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





alg = RandomForestClassifier()
param_grid = {'n_estimators':[5,7,10,12,16,24,35,47],'max_depth': [1,2,3,4,5,6,7,8,9,10],'min_samples_leaf':[1,2,4,6,8,10,12]}
grid = GridSearchCV(alg,param_grid=param_grid,cv=5)
grid.fit(X_train,y_train)
print(grid.best_params_)
# alg.fit(X_train, y_train)
# y_predicted = alg.predict(X_test)
# fpr, tpr, threshold = roc_curve(y_test, y_predicted, pos_label=1)
# listaauc.append(auc(fpr, tpr))
# validation.append(alg.score(X_train, y_train))
# test.append(alg.score(X_test, y_test))

# plt.show(plt.plot(range(1,50), test,range(1,50), validation))
