# -*- coding: utf-8 -*-
"""

reproduce error 


Created on Mon Nov 12 20:40:33 2018

@author: Germayne
"""

import lightgbm as lgb
import pandas as pd 
import numpy as np


# import 

X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')


y_train = X_train['target']
y_test = X_test['target']
X_train.drop(['target'], axis = 1, inplace = True)
X_test.drop(['target'], axis = 1, inplace = True)


# train 

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)


params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": 1,
    "min_data": 100,
    "boost_from_average": False
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=100)

prediction = model.predict(X_test)

# write out 

output = pd.DataFrame()
output['prediction'] = prediction

output.to_csv('result.csv', index = False)