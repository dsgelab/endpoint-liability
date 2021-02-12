#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################################################################
###########################################################################
Created on Mon Jan 04 2021

@author: Lisa Eick

Main script for executing all code at once
###########################################################################
###########################################################################
"""

import joblib
import os
from timeit import default_timer as timer

##############################################################################
############# Setting of paths and important variables #######################
##############################################################################
#input directory of your Code
codedir="/home/leick/Documents/AndreaGanna/Code/endpoint-liability/Python-Code"
endpointPath="/home/leick/Documents/AndreaGanna/Data/newFake/fake_endpoints_sub.csv"
pillPath="/home/leick/Documents/AndreaGanna/Data/newFake/fake_cum_pills_sub.csv"
#tree pic will be saved here
picPath=codedir + "/output"
#If you want a binary prediction set True alse False
binary= True
os.chdir(codedir)

##############################################################################
################## preparing Data and measuring time  ########################
##############################################################################
#START MY TIMER
start = timer()

#imports preped Data from DataPrep
import DataPrep as dataPrep
learnData=dataPrep.dataPrep(endpointPath, pillPath, binary)

#STOP MY TIMER
mergetimer=timer() - start
print(timer() - start, "s") # in seconds
 


#shortcut for already calculated Table
#learnData=pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table_small.csv")


##############################################################################
############# Extract endpoint data and creating modell ######################
############ Modell: xgbboost Hyperparaopt: Bayesian opt #####################
##############################################################################
#imports trained modell from ML-DecTree
import MLDecTree_bayesian_opt as xgbTree
#sets the endpoint of interest
endpoint="I9_STR_EXH"
delCol=["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY"]
#discards coloumns with high correlation to endpoint
corrValue=0.995
#final dataprep and modell training
accuracy, treeModell, corrDropList, X_test, y_test=xgbTree.MLdecTree(learnData, picPath, endpoint, delCol, corrValue, binary)


##############################################################################
################## getting plots and information about #######################
############ the accuracy and other validation of the modell #################
##############################################################################
import ModellView as mv
mv.modelView(treeModell, codedir, X_test, y_test)


#save model
joblib.dump(treeModell, picPath + "/EndpointModell.dat") 

#load saved model
#treeModell = joblib.load(codedir + "/EndpointModell.dat")