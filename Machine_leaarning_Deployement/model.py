#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:39:32 2022

@author: karthik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pickle


df=pd.read_csv("pudding.csv")
df.head()
df.dropna(inplace=True)
statuses=pd.get_dummies(df['status'])
statuses.head()
df=pd.concat([df,statuses],axis=1)
df.drop(['uuid','accountname','projectname','status','startdate','enddate','QueueDate','StartDate','PlanningDate','ImplementationDate','TestingDate','planning_decimal','implementation_decimal','complete'],axis=1,inplace=True)
#Train data
X=df.drop('result',axis=1)
y=df['result']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
prediction=logmodel.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)

# Saving model to disk
pickle.dump(logmodel, open('model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))