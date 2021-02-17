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


tabledir='/home/leick/Documents/AndreaGanna/Data/OldFake/FINNGEN_ENDPOINTS_DF6_public1.xlsx'
endpoint="I9_STR_EXH"

##############################################################################
############# Rename all htc code to simple OR expressions ###################
##############################################################################
def htc10clear (endpointInfo):
    reg = re.compile('\[\w+\|\w+\]')
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
                stempList=[]
                for i in range(len(tempList)):
                    #print(tempList[i])
                    if len(tempList[i]) == 2 and ("]" in tempList[i]) :
                        #print(tempList[i])
                        back=tempList[i].split(r']')[0]
                        front=tempList[i-1].split(r'[')
                        tempList[i]=front[0]+front[1]
                        tempList[i-1]=front[0]+back
                endpointInfo.loc[num, "HD_ICD_10"]="|".join(tempList) 
            #gets all XX[!-?][!-?] and solves them to XX?!|XX!!|XX??|XX!?
            if "][" in htcID:
                #print(htcID)
                tempList=list(filter(None, re.split('\[|\]', htcID)))
                i=0
                stempList=[]
                while i < len(tempList):
                    if "-" in tempList[i]:
                        #print(tempList[i])
                        for j in range(int(tempList[i][0]), int(tempList[i][len(tempList[i])-1])+1):
                            for k in range(int(tempList[i+1][0]), int(tempList[i+1][len(tempList[i+1])-1])+1):
                                idCode=str(tempList[i-1])+str(j)+str(k)
                                stempList.append(idCode)
                        i=i+1
                    i=i+1
                endpointInfo.loc[num, "HD_ICD_10"]="|".join(stempList) 
 #Done in two for loops because if not there where strange interferences between the if statements
    for num in range(endpointInfo.shape[0]):
        htcID=endpointInfo.loc[num, "HD_ICD_10"]#.reset_index(drop=True)[0]
        if not htcID is np.nan:
            if "[" in htcID:
                htcID=htcID.split("|")
                stempList=[]
                for tempID in htcID:
                    if "[" in tempID and "-" in tempID:
                        tempList=list(filter(None, re.split('\[|\]', tempID)))
                        #gets all XX[!-?] and solves them to XX?|XX*|XX!
                        if len(tempList) < 3:
                            for j in range(int(tempList[1][0]), int(tempList[1][len(tempList[1])-1])+1):
                                idCode=str(tempList[0])+str(j)
                                stempList.append(idCode)
                        #gets all XX[!-?]mm and solves them to XX?mm|XX*mm|XX!mm
                        else:
                            for j in range(int(tempList[1][0]), int(tempList[1][len(tempList[1])-1])+1):
                                idCode=str(tempList[0])+str(j)+tempList[2]
                                stempList.append(idCode)
                    else:
                        #gets all XX[145] and solves them to XX1|XX4|XX5
                        if "[" in tempID:
                            tempList=list(filter(None, re.split('\[|\]', tempID)))
                            for j in range(0, int(len(tempList[1]))):
                                idCode=tempList[0]+tempList[1][j]
                                stempList.append(idCode)
                        #gets all the rest
                        else:
                            stempList.append(tempID)  
                    #print(stempList)
                endpointInfo.loc[num, "HD_ICD_10"]="|".join(stempList) 
    return endpointInfo

##############################################################################
############# Get all Children of Endpoint of interest #######################
##############################################################################

def getAllChild (endpoint, childList, endpointInfo):
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
            getAllChild(child, childList, endpointInfo)
            #print(childList)
    #return childList

##############################################################################
############## Get all Parents of Endpoint of interest #######################
##############################################################################

def getAllParents (endpoint, parentList, colsearch, endpointInfo):
    ######get all direct parents#######
    parents=endpointInfo.dropna(subset=[colsearch]).reset_index(drop=True)
    parents1=parents[parents[colsearch].str.endswith("|" + endpoint)]["NAME"].tolist()
    parents5=parents[parents[colsearch].str.endswith("|" + endpoint + "|")]["NAME"].tolist()
    parents2=parents[parents[colsearch].str.startswith(endpoint + "|" )]["NAME"].tolist()
    parents3=parents[parents[colsearch].str.contains("\|" + endpoint + "\|" )]["NAME"].tolist()
    parents4=parents[~parents[colsearch].str.contains("\|" )]  
    parents4=parents4[parents4[colsearch].str.fullmatch(endpoint)]["NAME"].tolist()
    parents=parents1+parents2+parents3+parents4+parents5
    if parents:
        for parent in parents:
            parentList.append(parent)
            getAllParents(parent, parentList, colsearch, endpointInfo)

##############################################################################
############## Get all linked of Endpoint of interest #######################
##############################################################################

def getAllLinked (endpoint, linkedList, endpointInfog):
    endpointInfog=endpointInfog.dropna(subset=['HD_ICD_10']).reset_index(drop=True)
    #get the rowindex of endpoint of interest (eoi)
    rowIndex=endpointInfog.index[endpointInfog["NAME"] == endpoint]
    if not rowIndex.empty:
        for htc in endpointInfog.loc[rowIndex,'HD_ICD_10'].reset_index(drop=True)[0].split("|"):
            if htc and (not htc is np.nan):
                if len(htc) > 3:
                    htc=htc[0:3]
                linkedList= linkedList + endpointInfog[endpointInfog["HD_ICD_10"].str.contains(htc)]["NAME"].tolist()     
    linkedList = list(dict.fromkeys(linkedList))
    return linkedList





def getAllRealatedEndpoints (tabledir, endpoint):
    endpointInfo = pd.read_excel(tabledir)

    endpointInfog=endpointInfo.copy()
    endpointInfog=htc10clear(endpointInfog)
    
    childList=[]
    getAllChild(endpoint, childList, endpointInfo)
    
    parentList=[]
    getAllParents(endpoint, parentList, 'INCLUDE', endpointInfo)
    
    linkedList=[]
    linkedList = getAllLinked (endpoint, linkedList, endpointInfog)
    
    return parentList, childList, linkedList








