#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:12:34 2021

@author: leick

Main script for executing all code at once
"""
import os

#input directory of your Code
codedir="/home/leick/Documents/AndreaGanna/Code/endpoint-liability/Python-Code"
endpointPath="/home/leick/Documents/AndreaGanna/Data/newFake/fake_endpoints_sub.csv"
pillPath="/home/leick/Documents/AndreaGanna/Data/newFake/fake_cum_pills_sub.csv"
#tree pic will be saved here
picPath=codedir


os.chdir(codedir)

#imports preped Data from DataPrep
import DataPrep as dataPrep
learnData=dataPrep.dataPrep(endpointPath, pillPath)

#imports trained modell from ML-DecTree
import MLDecTree as xgbTree
#sets the endpoint of interest
endpoint="stroke"
delCol=["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY"]
#final dataprep and modell training
accuracy, treeModell=xgbTree.MLdecTree(learnData, picPath, endpoint, delCol)
