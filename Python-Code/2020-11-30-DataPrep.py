#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:23:50 2020

@author: leick
"""
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.setrecursionlimit(10000000)


#purchaseTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_purchases_sub.csv")
#packDrugTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_cum_packages_sub.csv")


def dataPrep():
    #loading in DataTables
    endpointTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_endpoints_sub.csv")
    pillsDrugTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_cum_pills_sub.csv")
    
    #replacing male and female with numeric identifier and make coloumn numeric
    endpointTable["SEX"]=endpointTable["SEX"].replace({"male":0,"female":1})
    
    #Getting all Names of columns i wish to drop using _NEVT
    collist=endpointTable.filter(like='_NEVT').columns
    endpointList = [i.split('_NEVT')[0] for i in collist] 
    
    #drop all unneccary columns
    clearTable=endpointTable.drop(endpointList, axis=1)
    clearTable=clearTable.drop(['BL_AGE'], axis=1)

    #parse everything possible tu numeric
    clearTable=clearTable.apply(pd.to_numeric)

    #drops all ages not related to an endpoint event
    for col in endpointList:
        trueT=clearTable[[col+"_NEVT"]]>0
        print(col)
        for i in range(trueT.shape[0]):
            if not trueT.iloc[i,0]:
                clearTable.iloc[i,clearTable.columns.get_loc(col+"_AGE")]=0 
                
    #merge Pill and Cleartable with each other
    mergedDf = clearTable.merge(pillsDrugTable, on='FINNGENID')
    mergedDf = mergedDf.fillna(0)
    
    #drop all columns which just contain 0
    mergedDf1=mergedDf.replace(0,np.nan).dropna(axis=1,how="all")
    mergedDf1 = mergedDf1.fillna(0)

#output if it is possible for avoiding calculating those Steps over and over again
#    mergedDf1.to_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table.csv")
#    Test= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table.csv")

