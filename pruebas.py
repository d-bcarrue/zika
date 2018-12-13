#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Linear, Bayesian, Multilayer Perceptron (Neural Nets), SVM (super vector
#machine), Random Forest, Decision Table
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import roc_curve,auc,confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
# LDC:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ldc = LinearDiscriminantAnalysis()

# Bayesian:
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# Multilayer Perceptron
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()

# SVM
from sklearn.svm import SVC
svm = SVC()

# Random Forest
from sklearn.ensemble import RandomForestClassifier 
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

from sklearn.model_selection import cross_val_score

score_ldc = cross_val_score(ldc, Xdata, Ydata, cv=5)
score_gnb = cross_val_score(gnb, Xdata, Ydata, cv=5)
score_mlp = cross_val_score(mlp, Xdata, Ydata, cv=5)
score_svm = cross_val_score(svm, Xdata, Ydata, cv=5)
score_rndf = cross_val_score(rndf, Xdata, Ydata, cv=5)
import numpy as np
print("LDC:",np.mean(score_ldc),np.std(score_ldc))
print("GNB:",np.mean(score_gnb),np.std(score_gnb))
print("MLP:",np.mean(score_mlp),np.std(score_mlp))
print("SVM:",np.mean(score_svm),np.std(score_svm))
print("RNDF:",np.mean(score_rndf),np.std(score_rndf))

