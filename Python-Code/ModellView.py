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
import eli5

#model=treeModell

def modelView (model, codedir, X_test, y_test):
    picpath=codedir + "/output"

    ##############################################################################
    ################ creating a Calibration plot and saving it ###################
    ##############################################################################
    proba = model.predict_proba(X_test) 
    mpv, fop=calibration_curve(y_true=y_test, y_prob=proba[:,1], n_bins=10)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    plt.savefig(picpath + '/calimatrix', format = "png", bbox_inches='tight')
    plt.clf() 
        
    ##############################################################################
    ######################## plot feature importance #############################
    ##############################################################################
    xgb.plot_importance(model, max_num_features = 20)
    plt.savefig(picpath + '/featurcountplot', format = "png", bbox_inches='tight')
    plt.clf()
    test=model.get_booster().get_score(importance_type='weight')
    test2=pd.DataFrame.from_dict(test,orient='index',columns=["featureimp"]).sort_values("featureimp", ascending=False)
    
    
    ##############################################################################
    ################# chaosgame plot feature importance ########################
    ##############################################################################
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False, plot_type="bar", max_display=30)
    f = plt.gcf()
    f.savefig(picpath + '/featureimportplot', format = "png", bbox_inches='tight')
    plt.clf() 

    ##############################################################################
    ################# permutation plot feature importance ########################
    ##############################################################################
    """
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_test, y_test,
                                n_repeats=5,
                               random_state=42,
                               n_jobs=-1)
 
    sorted_idx = r.importances_mean.argsort()
    plt.barh(X_test.columns[sorted_idx][:21], r.importances_mean[sorted_idx][:21])
    plt.xlabel("Permutation Importance")
    plt.savefig(picpath + '/test', format = "png", bbox_inches='tight')
    plt.clf() 
    """
    ##############################################################################
    ######################## table feature importance ############################
    ##############################################################################    
    kkk=model.feature_importances_
    feat_imp_df = pd.DataFrame(kkk, index=X_test.columns, columns=["featureimp"])
    feat_imp_df=feat_imp_df.sort_values("featureimp", ascending=False)
    
    ##############################################################################
    ######Confusion plot (makes sense when the value is binary classified) #######
    ##############################################################################
    conf = plot_confusion_matrix(model,
                          X_test,
                          y_test,
                          display_labels=["Have no stroke", "Have a stroke"])

    plt.savefig(picpath + '/confmatrix', format = "png", bbox_inches='tight')
    plt.clf() 


    ##############################################################################
    ################### Density plot for probality prediction ####################
    ##############################################################################

    df = pd.DataFrame(data=proba, columns=["col1", "col2"])
    strproba = df.iloc[:, 1].to_numpy()
    sb.set_style("whitegrid")  # Setting style(Optional)
    plt.figure(figsize = (10,5)) #Specify the size of figure we want(Optional)
    sb.distplot(strproba,  bins = 20, kde = True, color = 'teal', 
                kde_kws=dict(linewidth = 4 , color = 'black'))
    plt.savefig(picpath + '/densityplot', format = "png", bbox_inches='tight')
    plt.clf() 
    
    ##############################################################################
    ################### Output probality prediction Table ####################
    ##############################################################################
    probaoutput = pd.DataFrame(strproba, index=list(X_test.index))
    probaoutput.to_csv(codedir + "/probaoutput/ProbabilityTableTest.csv")

    
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
    plt.savefig(picpath + '/Roc curve', format = "png", bbox_inches='tight')
    plt.clf() 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    