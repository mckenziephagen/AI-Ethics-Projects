# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:ai-ethics-env]
#     language: python
#     name: conda-env-ai-ethics-env-py
# ---

from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

y_label = 'Strength_Unadj'

reg = linear_model.LassoCV()

label = pd.read_csv('unrestricted_mphagen_1_27_2022_20_50_7.csv')
label.rename(columns={ label.columns[0]: "Subject" }, inplace = True)

demos = pd.read_csv("demographics.csv", index_col='Subject')

X = pd.read_csv(f'HCP_PTN1200/netmats/3T_HCP1200_MSMAll_d15_ts2/netmats1.txt', 
                sep=' ', header=None ) 
sub_ids = pd.read_csv('subjectIDs.txt', header=None)
X['Subject'] = sub_ids
X = X.set_index('Subject')
X = X.merge(demographics[["Race"]], how='left', on='Subject')
label =label.loc[label['Subject'].isin(X.index), ['Subject', y_label]].dropna()
label = label.set_index('Subject')
X = X.loc[X.index.isin(label.index)]
x_train, x_test, y_train, y_test = train_test_split(X, label.loc[:, y_label], test_size=0.2)
lass = reg.fit(x_train.iloc[:,:-1], y_train)
lass.score(x_test.iloc[:,:-1], y_test)


for ii in range(1,100): 
    x_train, x_test, y_train, y_test = train_test_split(X, label.loc[:, y_label], test_size=0.2)
    lass = reg.fit(x_train.iloc[:,:-1], y_train)
    print("total:", lass.score(x_test.iloc[:,:-1], y_test))
    print("non-white:" , lass.score(x_test.loc[x_test['Race'] == 0, 0:224], y_test.loc[y_test.index.isin(x_test.loc[x_test['Race'] == 0].index)]) ) 
    print("white:", lass.score(x_test.loc[x_test['Race'] == 1, 0:224], y_test.loc[y_test.index.isin(x_test.loc[x_test['Race'] == 1].index)]) )
    print(" ")


