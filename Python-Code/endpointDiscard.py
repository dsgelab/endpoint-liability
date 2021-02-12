#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################################################################
###########################################################################
Created on Mon Feb 03 2021

@author: Lisa Eick

Discarding related endpoints from Data
###########################################################################
###########################################################################
"""

import pandas as pd
import numpy as np
import re
import fnmatch

tabledir="/home/leick/Documents/AndreaGanna/Data/OldFake"
endpointInfo= pd.read_excel(tabledir + "/FINNGEN_ENDPOINTS_DF6_public1.xlsx")
endpoint="III_BLOOD_IMMUN"
F5_INSOMNIA
COPD_ASTHMA_INFECTIONS
I9_STR_EXH
COPD_CVDMETABOCOMORB
ABPA
III_BLOOD_IMMUN
I9_HEARTFAIL
Z21_ADDIT_CODES_LOCAT_DEFECT_INJURY_ILLNE
test="F5_SLEEPWAKEE"
##############################################################################
############# Get all Children of Endpoint of interest #######################
##############################################################################

def getAllChild (endpoint, childList):
    #get the rowindex of endpoint of interest (eoi)
    rowIndex=endpointInfo.index[endpointInfo["NAME"] == endpoint]
    #get all direct children
    firstChildren=endpointInfo.loc[rowIndex, "INCLUDE"].reset_index(drop=True)
    #check if there are children 
    if not firstChildren[0] is np.nan:
        firstChildren=firstChildren[0].split("|")
        #get all children of children 
        for child in firstChildren:
            childList.append(child)
            getAllChild(child, childList)
            #print(childList)
    #return childList

##############################################################################
############## Get all Parents of Endpoint of interest #######################
##############################################################################

def getAllParents (endpoint, parentList):
    ######get all direct children#######
    parents=endpointInfo.dropna(subset=['INCLUDE']).reset_index(drop=True)
    parents1=parents[parents['INCLUDE'].str.endswith("|" + endpoint)]["NAME"].tolist()
    parents5=parents[parents['INCLUDE'].str.endswith("|" + endpoint + "|")]["NAME"].tolist()
    parents2=parents[parents['INCLUDE'].str.startswith(endpoint + "|" )]["NAME"].tolist()
    parents3=parents[parents['INCLUDE'].str.contains("\|" + endpoint + "\|" )]["NAME"].tolist()
    parents4=parents[~parents['INCLUDE'].str.contains("\|" )]  
    parents4=parents4[parents4['INCLUDE'].str.fullmatch(endpoint)]["NAME"].tolist()
    parents=parents1+parents2+parents3+parents4+parents5
    if parents:
        for parent in parents:
            parentList.append(parent)
            getAllParents(parent, parentList)

##############################################################################
############## Get all linked of Endpoint of interest #######################
##############################################################################

def getAllLinked (endpoint, parentList):
    #general 
    s.str.replace(r'[^(]*\(|\)[^)]*', '')
    #get the rowindex of endpoint of interest (eoi)
    rowIndex=endpointInfo.index[endpointInfo["NAME"] == "CD2_INSITU_DIGESTIVE_NOS"]    
    linked=endpointInfo.loc[rowIndex, "HD_ICD_10"].reset_index(drop=True)[0].split("|")
    for icdcode in linked:
        if "[" in icdcode:
            foo=list(filter(None, re.split('\[|\]|', icdcode)))
    parents=endpointInfo.dropna(subset=['INCLUDE']).reset_index(drop=True)
    parents1=parents[parents['INCLUDE'].str.endswith("|" + endpoint)]["NAME"].tolist()
    parents5=parents[parents['INCLUDE'].str.endswith("|" + endpoint + "|")]["NAME"].tolist()
    parents2=parents[parents['INCLUDE'].str.startswith(endpoint + "|" )]["NAME"].tolist()
    parents3=parents[parents['INCLUDE'].str.contains("\|" + endpoint + "\|" )]["NAME"].tolist()
    parents4=parents[~parents['INCLUDE'].str.contains("\|" )]  
    parents4=parents4[parents4['INCLUDE'].str.fullmatch(endpoint)]["NAME"].tolist()
    parents=parents1+parents2+parents3+parents4+parents5
    if parents:
        for parent in parents:
            parentList.append(parent)
            getAllParents(parent, parentList)





childList=[]
getAllChild(endpoint, childList)

parentList=[]
getAllParents(endpoint, parentList)
reg = re.compile('\[.\|.\]')
endpointInfo["HD_ICD_10"]=endpointInfo["HD_ICD_10"].replace({"%":"", "&":"|"}, regex=True)
for num in range(endpointInfo.shape[0]):
    htcID=endpointInfo.loc[num, "HD_ICD_10"]#.reset_index(drop=True)[0]
    if not htcID is np.nan:
        #changes "$!$" to nan
        if "$!$" in htcID:
            endpointInfo.loc[num, "HD_ICD_10"]=np.nan
        #gets all XX[!|?] and solves them to XX?|XX!
        if bool(re.search(reg, htcID)):
            tempList=htcID.split(r'|')
            for i in range(len(tempList)):
                if len(tempList[i]) == 2:
                    back=tempList[i].split(r']')[0]
                    front=tempList[i-1].split(r'[')
                    tempList[i]=front[0]+front[1]
                    tempList[i-1]=front[0]+back
                    endpointInfo.loc[num, "HD_ICD_10"]="|".join(tempList) 
        #gets all XX[!-?] and solves them to XX?|XX!
        if bool(re.search(reg, htcID)):
            tempList=htcID.split(r'|')
        
                    
print(tempList[i])
            htcID.split(r'|')

re.findall(re.compile('\[.\|.\]'), htcID)


CD2_INSITU_DIGESTIVE_NOS




reg = re.compile('\[.\|.\]')

match = bool(re.search(reg, htcID))






