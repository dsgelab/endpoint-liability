#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:23:50 2020

@author: leick

Visualization of data
"""
import pandas as pd
import seaborn as sns
import sys
sys.setrecursionlimit(10000000)


#purchaseTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_purchases_sub.csv")
#packDrugTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_cum_packages_sub.csv")
endpointTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_endpoints_sub.csv")
#pillsDrugTable= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/fake_cum_pills_sub.csv")
    



listi=endpointTable.columns
looki= endpointTable.columns[12]
endpointTable=endpointTable.astype(float)

cut=endpointTable.iloc[:,15:]
cut=cut.fillna(-20)
sns.clustermap(cut, row_cluster=False)