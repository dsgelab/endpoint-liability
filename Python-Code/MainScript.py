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
codedir="/home/jsjukara/lisa/endpoint-liability/Python-Code"
endpointPath="/home/jsjukara/finngen_data/r6_data/real_endpoints_int_sub.csv"
pillPath="/home/jsjukara/finngen_data/r6_data/real_cum_pills_int_sub.csv"
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
delCol=["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY", "I9_CEREBVASC", "FG_CEREBVASC", "C_STROKE", "I9_CVD_HARD", "I9_CVD", "ASTHMA_CVDMETABOCOMORB", "COPD_CVDMETABOCOMORB" ,"ILD_CVDMETABOCOMORB", "ILD_COMORB", "NEURODIS", "FG_CVD", "RA_CO_MORB", "C_PSOR_ARTHR_COMORB", "SLE_COMORB", "DM_COMORB", "ASTHMA_CVDMETABOCOMORB", "ASTHMA_COMORB", "RA_CO_MORB", "RHEUMA_COMORB", "C_PSOR_ARTHR_COMORB", "SLE_COMORB", "IBD_COMORB", "COPD_MULTIMORB", "MULTIMORB", "NEURO_COMORBIDITIES", "COPD_COMORB"]

#discards coloumns with high correlation to endpoint
corrValue=0.995
#final dataprep and modell training
accuracy, treeModell, corrDropList, X_test, y_test=xgbTree.MLdecTree(learnData, picPath, endpoint, delCol, corrValue, binary)


##############################################################################
################## getting plots and information about #######################
############ the accuracy and other validation of the modell #################
##############################################################################
import ModellView as mv
mv.modelView(treeModell, picPath, X_test, y_test)


#save model
joblib.dump(treeModell, picPath + "/EndpointModell.dat") 

#load saved model
#treeModell = joblib.load(codedir + "/EndpointModell.dat")