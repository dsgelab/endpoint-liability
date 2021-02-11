#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################################################################
###########################################################################
Created on Mon Feb 10 2021

@author: Lisa Eick

gets plots and other graphs helping for giving insights of the modell
###########################################################################
###########################################################################
"""
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.calibration import calibration_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



def modelView (model, picpath, X_test, y_test):
    ##############################################################################
    ################ creating a Calibration plot and saving it ###################
    ##############################################################################
    proba = model.predict_proba(X_test) 
    mpv, fop=calibration_curve(y_true=y_test, y_prob=proba[:,1], n_bins=10)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    plt.savefig(picpath + '/calimatrix', format = "png")
    plt.clf() 
        
    
    ##############################################################################
    ######################## plot feature importance #############################
    ##############################################################################
    xgb.plot_importance(model, max_num_features = 20)
    plt.savefig(picpath + '/featurimportanceplot', format = "png")
    plt.clf()
 
    
    ##############################################################################
    ######Confusion plot (makes sense when the value is binary classified) #######
    ##############################################################################
    conf = plot_confusion_matrix(model,
                          X_test,
                          y_test,
                          display_labels=["Have no stroke", "Have a stroke"])

    plt.savefig(picpath + '/confmatrix', format = "png")
    plt.clf() 


    ##############################################################################
    ################### Density plot for probality prediction ####################
    ##############################################################################
    proba = model.predict_proba(X_test) 
    mpv, fop=calibration_curve(y_true=y_test, y_prob=proba[:,1], n_bins=10)

    df = pd.DataFrame(data=proba, columns=["col1", "col2"])
    data = df.iloc[:, 0].to_numpy()
    sb.set_style("whitegrid")  # Setting style(Optional)
    plt.figure(figsize = (10,5)) #Specify the size of figure we want(Optional)
    sb.distplot(data,  bins = 20, kde = True, color = 'teal', 
                kde_kws=dict(linewidth = 4 , color = 'black'))
    plt.savefig(picpath + '/densityplot', format = "png")
    plt.clf() 

    ##############################################################################
    ###### printing out first of the xgb Tree calculated and make it pretty ######
    ##############################################################################
    #bst = model.get_booster()
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
    
    ##############################################################################
    ############################ Roc-curve print out #############################
    ##############################################################################
    # keep probabilities for the positive outcome only
    lr_probs = proba[:, 1]
    
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]
    
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # save the plot
    plt.savefig(picpath + '/Roc curve', format = "png")
    plt.clf() 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    