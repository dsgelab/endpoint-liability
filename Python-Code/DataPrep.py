#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################################################################
###########################################################################
Created on Mon Nov 30 2020

@author: Lisa Eick

Prepares Data for Modell input
###########################################################################
###########################################################################
"""
import numpy as np
import pandas as pd
#if calculation breaks of because of time limit
#import sys
#sys.setrecursionlimit(10000000)



#purchaseTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_purchases_sub.csv")
#packDrugTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_cum_packages_sub.csv")


def dataPrep(endpointPath, pillPath, binary):
    #path if not set
    #endpointPath="/home/leick/Documents/AndreaGanna/Data/newFake/fake_endpoints_sub.csv"
    #for string ID input
    #endpointPath="/home/leick/Documents/AndreaGanna/Data/newFake/fake_endpoints_sub_strID.csv"
    #pillPath="/home/leick/Documents/AndreaGanna/Data/newFake/fake_cum_pills_sub1.csv"

    #loading in DataTables
    endpointTable= pd.read_csv(endpointPath, low_memory=False)
    pillsDrugTable= pd.read_csv(pillPath)
    
    #replacing male and female with numeric identifier and make coloumn numeric
    endpointTable["SEX"]=endpointTable["SEX"].replace({"male":0,"female":1})
    
    #Getting all Names of columns i wish to drop using _NEVT
    endpointList=endpointTable.filter(like='_NEVT').columns
    endpointList = [i.split('_NEVT')[0] for i in endpointList] 
    
    #drop all unneccary columns
    clearTable=endpointTable.drop(endpointList, axis=1)
    clearTable=clearTable.drop(['BL_AGE'], axis=1)

    #merge Pill and Cleartable with each other
    #clearTable = clearTable.merge(pillsDrugTable, on='FINNGENID')
    #clearTable = clearTable.drop(['FINNGENID'], axis=1)
   
    #alternativ merge using join
    clearTable=clearTable.set_index('FINNGENID').sort_index()
    pillsDrugTable=pillsDrugTable.set_index('FINNGENID').sort_index()
    clearTable = clearTable.join(pillsDrugTable, how='outer') 
    
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
        else:
            if col.split('_NEVT')[0] in endpointList:
                trueT=np.array(clearTable[col]>0)
                #print(col)
                a = np.array(clearTable[col.split('_NEVT')[0]+"_AGE"])
                a = np.where(trueT, a, a*0)
                pdNump=pd.DataFrame(a, index=clearTable.index)
                clearTable[col.split('_NEVT')[0]+"_AGE"]=pdNump
                
    #Drop Coloumns with lower than 0.005 cases
    #clearTable2=clearTable1.loc[:, (clearTable1==0).mean() + mist=(endpointTable.isnull()).mean() < .995]#TODO Missingvalues stay in table
    clearTable = clearTable.replace(0,np.nan).dropna(thresh=clearTable.shape[0]*0.005, axis=1)
    clearTable = clearTable.fillna(0)
    #2988 mergclearTable
    
    #drop all columns which just contain one kind of value using dropCol
    dropList = [i.split('_NEVT')[0] for i in dropList] 
    mask_pattrn = '|'.join(dropList)
    if mask_pattrn:
        clearTable = clearTable[clearTable.columns.drop(list(clearTable.filter(regex=mask_pattrn)))]

    #for binary prediction switches nevt to binary
    if binary is True:
        nevtColumn = [s for s in clearTable.columns if "nevt" in s.lower()]
        for colu in nevtColumn:
            clearTable[colu].values[clearTable[colu].values > 1] = 1
        
    return clearTable
    
#output if it is possible for avoiding calculating those Steps over and over again
#    clearTable.to_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table_small.csv")
#    Test= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table.csv")

