#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:00:34 2020

@author: leick
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import importlib
import math
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, make_scorer, confusion_matrix, plot_confusion_matrix


#input directory of your Code
codedir="/home/leick/Documents/AndreaGanna/Code/endpoint-liability/Python-Code"
os.chdir(codedir)

#imports preped Data from DataPrep
import DataPrep as dataPrep

#Gets the prepared Data
#learnData= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table.csv")
#learnData=pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table_small.csv")
#endpoint="stroke"
#delCol =["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY"]
#matching=list(Test.filter(regex=mask_pattrn))

def MLdecTree (endpoint="stroke", delCol=["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY"]):
    #reads in processed Data from other function
    learnData=dataPrep.dataPrep()
    learnColumn=learnData.columns

    
    #correlates all the nevt columns with the target columns and saves the columns with high corr in list 
    matching = [s for s in learnColumn if endpoint in s.lower()]
    matching = [s for s in matching if "nevt" in s.lower()]
    corrDropCol=[]
    for i in range(learnData.shape[1]):
        if "nevt" in learnData.columns[i].lower():
            for match in matching:
                corrCo=learnData[match].corr(learnData.iloc[:,i], method='spearman')
                print(corrCo)
                if corrCo > 0.995 or corrCo < -0.995:
                    corrDropCol.append(learnData.columns[i])
 
    #drop all columns which are medicly too close related to endpoint 
    mask_pattrn = '|'.join(delCol)  
    learnData = learnData[learnData.columns.drop(list(learnData.filter(regex=mask_pattrn)))]

    #deletes all strongly corr columns
    list(set(corrDropCol)-set(matching))
    learnData=learnData.drop(corrDropCol, axis=1)
    
    #Splitting dependent and independent Variable y=result
    X = learnData.drop(matching[0], axis=1).copy()
    y = learnData[matching[0]].copy()
    
    #splitting Data in train and test Data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    #fitting xgbTree
    clf_xgb= xgb.XGBClassifier(objective="binary:logistic", missing=None, seed=42)
    clf_xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric="aucpr", eval_set=[(X_test, y_test)])
    
    
'''
Test how much % of data is positiv
    y_test[y_test > 0] = 1  
    sum(y_test)/len(y_test)
'''    
