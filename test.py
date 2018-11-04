#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" TP2

https://openclassrooms.com/fr/courses/4297211-evaluez-et-ameliorez-les-performances-dun-modele-de-machine-learning/4308241-mettez-en-place-un-cadre-de-validation-croisee
http://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold
"""

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.impute import SimpleImputer
import sklearn.preprocessing as skp
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

# from mlxtend.preprocessing import TransactionEncoder

# Récupère les données à partir du fichier et les insère dans un DataFrame
vote, meta = loadarff('labor.arff')
df = pd.DataFrame(vote)

# --------------------------
# NORMALISATION (Question 1)
# --------------------------
# Noms des colonnes numériques (cad à normaliser)
numeric_columns = [meta.names()[i] for i, t in enumerate(meta.types()) if t == 'numeric']
# Remplace les valeurs manquantes par la moyenne
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df[numeric_columns] = imp_mean.fit_transform(df[numeric_columns])
# Normalisation
df[numeric_columns] = skp.StandardScaler().fit_transform(df[numeric_columns])
# print(df)

# ----------
# Question 3
# ----------
# X = df.loc[:, df.columns != 'class']
X = df[numeric_columns]
y = df['class']

lst_classif = [
	DummyClassifier(strategy="most_frequent"),
	GaussianNB(),
	tree.DecisionTreeClassifier(),
	LogisticRegression(solver="liblinear"),
	svm.SVC(gamma='scale'),
]

lst_classif_names = [
	'Dummy',
	'Naive Bayes',
	'Decision tree',
	'Logistic regression',
	'SVM'
]

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))

accuracy_score(lst_classif,lst_classif_names,X,y)
confusion_matrix(lst_classif,lst_classif_names,X,y)

# ----------
# Question 4
# ----------
# Remplace les valeurs manquantes par la valeur la plus fréquente
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# print(df[:, df.columns not in numeric_columns])
# print(df.columns.all(numeric_columns))
