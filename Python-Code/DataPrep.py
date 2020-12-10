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
    endpointList=endpointTable.filter(like='_NEVT').columns
    endpointList = [i.split('_NEVT')[0] for i in endpointList] 
    
    #drop all unneccary columns
    clearTable=endpointTable.drop(endpointList, axis=1)
    clearTable=clearTable.drop(['BL_AGE'], axis=1)
    
    #merge Pill and Cleartable with each other
    clearTable = clearTable.merge(pillsDrugTable, on='FINNGENID')
    
    #parse everything possible tu numeric
    clearTable=clearTable.apply(pd.to_numeric)
    
    #goes through every column in endpoint data
    dropList=[]
    for col in clearTable.columns:
    #checks if all columns are numeric. prints out warning if not
        if not np.issubdtype(clearTable[col].dtypes, np.number):
            print("Warning: " + col + " is not numeric")
    #checks if there is just 1 kind of variable in this column. If so writes it into drop list
        if clearTable[col].unique().size < 2:
            dropList.append(col)
    #drops all ages in _AGES which are not related to an endpoint event in _NEVT 
        if col.split('_NEVT')[0] in endpointList:
            trueT=np.array(clearTable[col]>0)
            print(col)
            a = np.array(clearTable[col.split('_NEVT')[0]+"_AGE"])
            a = np.where(trueT, a, a*0)
            clearTable[col.split('_NEVT')[0]+"_AGE"]=pd.Series(a)
                  
    #Drop Coloumns with lower than 0.005 cases
    clearTable=clearTable.replace(0,np.nan).dropna(thresh=clearTable.shape[0]*0.005, axis=1)
    clearTable = clearTable.fillna(0)

    
    #drop all columns which just contain one kind of value using dropCol
    dropList = [i.split('_NEVT')[0] for i in dropList] 
    mask_pattrn = '|'.join(dropList)
    clearTable = clearTable[clearTable.columns.drop(list(clearTable.filter(regex=mask_pattrn)))]



    
    return clearTable
    
#output if it is possible for avoiding calculating those Steps over and over again
#    clearTable.to_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table_small.csv")
#    Test= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table.csv")

