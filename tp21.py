#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import tp21func

# Question 1
data = loadarff('labor.arff')
df = pd.DataFrame(data[0])

# determine class balance
i = 0
for item in df['class']:
    if item.decode('utf-8') == "'good'":
        i += 1
    else:
        i -= 1
print(i)

# Question 2
# 5 7 10 12-17 | 1 2 3 4 6 8 9 11
# print(df['duration'])
numeric_columns = ['duration',
          'wage-increase-first-year',
          'wage-increase-second-year',
          'wage-increase-third-year',
          'working-hours',
          'standby-pay',
          'shift-differential',
          'statutory-holidays']

mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df[numeric_columns] = mean.fit_transform(df[numeric_columns])
df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

X = df[numeric_columns]
y = df['class']

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

lst_classif = [dummycl, gmb, dectree, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression', 'SVM']

# Question 3
tp21func.accuracy_score(lst_classif, lst_classif_names, X, y)
# Question 4

categorical_columns = ['cost-of-living-adjustment',
                     'pension',
                     'education-allowance',
                     'vacation',
                     'longterm-disability-assistance',
                     'contribution-to-dental-plan',
                     'bereavement-assistance',
                     'contribution-to-health-plan',
                     'class']

freq = SimpleImputer(missing_values=b'?', strategy='most_frequent')
df[categorical_columns] = freq.fit_transform(df[categorical_columns])
#print(df)
df_dummies = pd.get_dummies(df, columns=categorical_columns)
print(df_dummies)

tp21func.accuracy_score(lst_classif, lst_classif_names, X, y)