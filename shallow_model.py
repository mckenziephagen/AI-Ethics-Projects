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

reg = linear_model.LassoCV()

label = pd.read_csv('cognition_scores.csv')
label.rename(columns={ label.columns[0]: "Subject" }, inplace = True)

demographics = pd.read_csv("race.csv", index_col='Subject').drop(columns='Unnamed: 0')

X = pd.read_csv(f'HCP_PTN1200/netmats/3T_HCP1200_MSMAll_d15_ts2/netmats1.txt', 
                sep=' ', header=None ) 
sub_ids = pd.read_csv('subjectIDs.txt', header=None)
X['Subject'] = sub_ids
X = X.set_index('Subject')
X = X.merge(demographics[["Race"]], how='left', on='Subject')
label =label.loc[label['Subject'].isin(X.index)]
X = X.loc[X.index.isin(label['Subject'])]
x_train, x_test, y_train, y_test = train_test_split(X, label.loc[:, 'CogTotalComp_Unadj'], test_size=0.2, random_state=42)
lass = reg.fit(x_train.iloc[:,:-1], y_train)
lass.score(x_test.iloc[:,:-1], y_test)

