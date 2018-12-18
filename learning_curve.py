#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# plotting the learning curve for Random Forest Classifier
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.utils import shuffle

df = pd.read_csv(os.path.join('ds_Zika.csv'))

# Under-sampling
nposit = len(df[df['Output'] == 1])
inegat = df[df.Output == 0].index
random_indices = np.random.choice(inegat, nposit, replace=False)
iposit = df[df.Output == 1].index
new_indices = np.concatenate([iposit, random_indices])
df = df.loc[new_indices]
df = shuffle(df).reset_index(drop=True)

X = df.drop('Output', axis=1)
y = df['Output']
# learning curve calculation
cv = ShuffleSplit(n_splits=100, test_size=0.2)
alg = RandomForestClassifier(n_estimators=100)

train_sizes = np.linspace(0.05, 1, num=20)
train_sizes, train_scores, test_scores = learning_curve(
        alg, X, y, cv=cv, train_sizes=train_sizes)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# plot
plt.figure()
plt.title('Learning curve RFC')
plt.xlabel('Training size')
plt.ylabel('Score')
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='b')
plt.plot(train_sizes, train_scores_mean, 'ro-', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'bo-', label='Test score')
plt.legend(loc='best')
plt.show()
