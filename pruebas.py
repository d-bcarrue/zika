#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Linear, Bayesian, Multilayer Perceptron (Neural Nets), SVM (super vector
# machine), Random Forest, Decision Table
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# K Neighbors
knb = KNeighborsClassifier()

# LDC:
ldc = LinearDiscriminantAnalysis()

# Bayesian:
gnb = GaussianNB()

# Multilayer Perceptron
mlp = MLPClassifier()

# SVM:
svm = SVC()

# Random Forest
rndf = RandomForestClassifier()

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

score_knb = cross_val_score(knb, Xdata, Ydata, cv=10)
score_ldc = cross_val_score(ldc, Xdata, Ydata, cv=10)
score_gnb = cross_val_score(gnb, Xdata, Ydata, cv=10)
score_mlp = cross_val_score(mlp, Xdata, Ydata, cv=10)
score_svm = cross_val_score(svm, Xdata, Ydata, cv=10)
score_rndf = cross_val_score(rndf, Xdata, Ydata, cv=10)
print((6*' ')+'mean Accuracy'+(6*' ')+'standard deviation')
print("KNB: ",np.mean(score_knb), np.std(score_knb))
print("LDC: ",np.mean(score_ldc), np.std(score_ldc))
print("GNB: ",np.mean(score_gnb), np.std(score_gnb))
print("MLP: ",np.mean(score_mlp), np.std(score_mlp))
print("SVM: ",np.mean(score_svm), np.std(score_svm))
print("RNDF:",np.mean(score_rndf), np.std(score_rndf))

