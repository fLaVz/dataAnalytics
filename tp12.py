#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from scipy.io.arff import loadarff
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np


data = loadarff('vote.arff')
# convert categorical values into one-hot vectors and ignore ? values
# corresponding to missing values
# ex: handicapped-infants=y -> [1,0], handicapped-infants=n -> [0,1], handicapped-infants=? -> [0,0]
df = pd.DataFrame(data[0])
vote_one_hot = pd.get_dummies(df)
vote_one_hot.drop(vote_one_hot.filter(regex='_\?$', axis=1).columns, axis=1, inplace=True)

# Question 1
# Les données manquantes sont représentées par des '?'
# Pour ne pas corrompre les resultas, on ne les prends pas en compte en attendant de trouver une meilleure solution

# Question 2
# 118 items 
frequent_itemsets = apriori(vote_one_hot, min_support=0.4)
#print(frequent_itemsets)

# Question 3
# 192
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
print(rules)


# Question 4
# Question 5 a finir