#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 09:46:59 2022

@author: demix9
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from machinelearning import *


data = pd.read_csv(r'/Users/demix9/Desktop/squark_automotive_CLV_training_data.csv',index_col = 0)
data.dropna(how = 'any',inplace = True)
DF= pd.DataFrame([])
DF['State'] = LabelEncoder().fit_transform(data['State'])
DF['Response'] = LabelEncoder().fit_transform(data['Response'])
DF['Coverage'] = LabelEncoder().fit_transform(data['Coverage'])
DF['Education'] = LabelEncoder().fit_transform(data['Education'])
DF['EmploymentStatus'] = LabelEncoder().fit_transform(data['EmploymentStatus'])
DF['Gender'] = LabelEncoder().fit_transform(data['Gender'])
DF['Income'] = data['Income'].astype('float').values
DF['Location Code'] = LabelEncoder().fit_transform(data['Location Code'])
DF['Marital Status'] = LabelEncoder().fit_transform(data['Marital Status'])
DF['Monthly Premium Auto'] = LabelEncoder().fit_transform(data['Monthly Premium Auto'])
DF['Monthly Premium Auto'] = data['Monthly Premium Auto'].values
DF['Months Since Last Claim'] = data['Months Since Last Claim'].values
DF['Months Since Policy Inception'] = data['Months Since Policy Inception'].values
DF['Number of Open Complaints'] = data['Number of Open Complaints'].values
DF['Number of Policies'] = data['Number of Policies'].values
DF['Policy Type'] = LabelEncoder().fit_transform(data['Policy Type'])
DF['Policy'] = LabelEncoder().fit_transform(data['Policy'])
DF['Renew Offer Type'] = LabelEncoder().fit_transform(data['Renew Offer Type'])
DF['Sales Channel'] = LabelEncoder().fit_transform(data['Sales Channel'])
DF['Total Claim Amount'] = data['Total Claim Amount'].values
DF['Vehicle Class'] = LabelEncoder().fit_transform(data['Vehicle Class'])
DF['Vehicle Size'] = LabelEncoder().fit_transform(data['Vehicle Size'])
DF['Customer Lifetime Value'] = data['Customer Lifetime Value'].values



a = DF.describe().T 


import matplotlib.pyplot as plt
cm = plt.cm.get_cmap('RdYlBu_r')
dfData = DF.corr().round(2)
import seaborn as sns

plt.subplots(figsize=(20, 20)) # 设置画面大小

#cm = plt.cm.get_cmap('RdYlBu')
m = sns.heatmap(dfData, annot=True, linewidths = 0.05,vmax=1, square=True, cmap=cm)
#cbar = plt.colorbar(m)
#cbar.set_label('$T_B(K)$',fontdict=font)
#cbar=m.colorbar(im,'bottom',size=0.2,pad=0.3,label='W*m-2')

#cbar.set_ticks(np.linspace(-1,1,0.1))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.savefig('./BluesStateRelation.png')
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
plt.show()

Y = DF['Customer Lifetime Value']
X = DF.drop('Customer Lifetime Value',axis=1)

from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state = 0)
names = ['Decision Tree', 'Linear Regression', 'Ridge','Lasso','SVR', 'KNN', 'RFR', 'Ada Boost', 
    'Gradient Boost', 'Bagging', 'Extra Tree','XGBoost']
ML = machinelearning(names[6],Xtrain,ytrain,Xtest,ytest)
performance =  ML.model_predict()



