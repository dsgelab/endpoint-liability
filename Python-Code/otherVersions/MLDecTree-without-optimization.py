#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:00:34 2020

@author: leick

final data preparation and modell learning
"""
import re
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, make_scorer, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import export_graphviz
import graphviz    
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RandomizedSearchCV
            
            
#Gets the prepared Data
#learnData= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table.csv")
#learnData=pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table_small.csv")
#endpoint="stroke"
#delCol =["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY"]
#matching=list(Test.filter(regex=mask_pattrn))

def MLdecTree (learnData, picpath, endpoint="I9_STR_EXH", delCol=["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY"], corrValue=0.995, binary=True):
    #reads in processed Data from other function
    learnColumn=learnData.columns
    
    #correlates all the nevt columns with the target columns and saves the columns with high corr in list 
    matching = [s for s in learnColumn if endpoint.lower() in s.lower()]
    endpointofInterest = [s for s in matching if "nevt" in s.lower()]
    corrDropCol=[]
    for colName in learnData.columns:
        #print(colName)
        if "nevt" in colName.lower():
            coreName=colName.split('_NEVT')[0]
            for match in matching:
                corrCo=learnData[match].corr(learnData[colName], method='spearman')
                if (corrCo > corrValue) or (corrCo < -corrValue):
                    #spike_cols = [col for col in learnColumn if coreName in col]
                    corrDropCol.extend([colName, coreName+"_AGE"])

    #setting the y for endpoint of interest
    y = learnData[endpointofInterest[0]].copy().to_numpy()
    y = y.astype(int)

    #drop all columns which are medicly too close related to endpoint 
    mask_pattrn = '|'.join(delCol) 
    if mask_pattrn:
        learnData1 = learnData[learnData.columns.drop(list(learnData.filter(regex=mask_pattrn)))]

    #deletes all strongly corr columns
    corrDropCol=list(set(corrDropCol)-set(matching))
    mask_pattrn = '|'.join(corrDropCol)  
    if mask_pattrn:
        learnData1 = learnData1[learnData1.columns.drop(list(learnData1.filter(regex=mask_pattrn)))]
    
    #Splitting dependent and independent Variable y=result
    mask_pattrn = '|'.join(matching)  
    if mask_pattrn:
        X = learnData1[learnData1.columns.drop(list(learnData1.filter(regex=mask_pattrn)))]
    else:
        X=learnData1

    
    #splitting Data in train and test Data set
    y=pd.Series(preprocessing.LabelEncoder().fit_transform(np.array(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


#    ka=pd.Series(y.unique()).sort_values()
#   , use_label_encoder=False
    #fitting xgbTree
#    clf_xgb= xgb.XGBClassifier(use_label_encoder=False) objective="multi:softmax", num_class = len(y.unique()))
#    clf_xgb.fit(X_train, y_train, eval_metric="merror", eval_set=[(X_test, y_test)])
    
    if binary is True:
        #to be modified: gamma, n_jobs(threads)
        #normal weight: scale_pos_weight= (y != 0).sum()/(y == 0).sum()
        model = XGBClassifier(base_score=0.5, booster="gbtree", colsample_bylevel=1, colsample_bynode=1, 
                          colsample_bytree=1, gamma=0.25, learning_rate=0.1, max_delta_step=0,
                          max_depth=6, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1, 
                          objective="binary:logistic", random_state=0, reg_alpha=0, reg_lambda=1,
                          scale_pos_weight=30, seed=None, subsample=1, verbosity=1) 
	#eval_metric:aucpr + auc + logloss
        model.fit(X_train, y_train, verbose=True, eval_metric="auc")
        
    else:
        lc = LabelEncoder() 
        lc = lc.fit(y)   
        model = XGBClassifier(base_score=0.5, booster="gbtree", colsample_bylevel=1, colsample_bynode=1, 
                          colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                          max_depth=6, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1, 
                          objective="reg:tweedie", random_state=0, reg_alpha=0, reg_lambda=1,
                          scale_pos_weight=(y == 0).sum()/(y != 0).sum(), seed=None, subsample=1, verbosity=1) 
        #eval_metric:rmse + tweedie-nloglik
        model.fit(X_train, y_train, verbose=True, eval_metric="tweedie-nloglik")

#The accuracy of the model is calculated and printed
    y_pred = model.predict(X_test) 
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions) 

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
 
#Confusion plot (makes sense when the value is binary classified)
    conf = plot_confusion_matrix(model,
                          X_test,
                          y_test,
                          display_labels=["Have no stroke", "Have a stroke"])

    plt.savefig(picpath + '/confmatrix', format = "png")
    
#Code for printing out the xgb Tree calculated and make it pretty    
    bst = model.booster()
    #for importance_type in ("weight","gain","cover","total_gain","total_cover"):
    #    print("%s: " % importance_type, bst.get_score(importance_type=importance_type))
    #next two section is to make visual adjustments
    node_params = {"shape": "box",
                   "style": "filled, rounded",
                   "fillcolor": "#78cbe"}
    leaf_params= {"shape" : "box",
                  "style" : "filled",
                  "fillcolor" : "#e48038"}
    #creates tree
    image = xgb.to_graphviz(model, num_trees=0, size="10,10",
                    condition_node_params=node_params,
                    leaf_node_params=leaf_params)
    
    #Set a different dpi (work only if format == 'png')
    image.graph_attr = {'dpi':'400'}
    #Saving the tree where the code is saved
    image.render(picpath + '/modellbild1', format = "png")

    return accuracy, model, corrDropCol
 
