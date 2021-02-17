#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################################################################
###########################################################################
Created on Mon Dec 07 2020

@author: Lisa Eick

final data preparation and modell learning
###########################################################################
###########################################################################
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from skopt.space import Real 
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
#import matplotlib.pyplot as plt
#from sklearn.calibration import calibration_curve
          

#Gets the prepared Data if not measured before
#learnData= pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table.csv")
#learnData=pd.read_csv("/home/leick/Documents/AndreaGanna/Data/newFake/2020-12-07-con_endpoint_drug_table_small.csv")
#endpoint="stroke"
#delCol =["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY"]
#matching=list(Test.filter(regex=mask_pattrn))

def MLdecTree (learnData, picpath, endpoint="I9_STR_EXH", delCol=["I9_STR_SAH","I9_SEQULAE", "I9_STR", "IX_CIRCULATORY"], corrValue=0.995):
    #reads in processed Data from other function
    learnColumn=learnData.columns
    
    #correlates all the nevt columns with the target columns and saves the columns with high corr in list 
    matching = [s for s in learnColumn if endpoint.lower() in s.lower()]
    endpointofInterest = [s for s in matching if "nevt" in s.lower()]
    corrDropCol=[]
    for colName in learnData.columns:
        #print(colName)
        if "nevt" in colName.lower():
            coreName=colName.split('_NEVT')[0]
            for match in matching:
                corrCo=learnData[match].corr(learnData[colName], method='spearman')
                if (corrCo > corrValue) or (corrCo < -corrValue):
                    #spike_cols = [col for col in learnColumn if coreName in col]
                    corrDropCol.extend([colName, coreName+"_AGE"])

    #setting the y for endpoint of interest
    y = learnData[endpointofInterest[0]].copy().to_numpy()
    y = y.astype(int)

    #drop all columns which are medicly too close related to endpoint 
    mask_pattrn = '|'.join(delCol) 
    if mask_pattrn:
        learnData1 = learnData[learnData.columns.drop(list(learnData.filter(regex=mask_pattrn)))]

    #deletes all strongly corr columns
    corrDropCol=list(set(corrDropCol)-set(matching))
    mask_pattrn = '|'.join(corrDropCol)  
    if mask_pattrn:
        learnData1 = learnData1[learnData1.columns.drop(list(learnData1.filter(regex=mask_pattrn)))]
    
    #Splitting dependent and independent Variable y=result
    mask_pattrn = '|'.join(matching)  
    if mask_pattrn:
        X = learnData1[learnData1.columns.drop(list(learnData1.filter(regex=mask_pattrn)))]
    else:
        X = learnData1

    
    #splitting Data in train and test Data set
    y=pd.Series(preprocessing.LabelEncoder().fit_transform(np.array(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    #parameter wich have to be optimized
    #bayesion opt and first modell fitting    
    #to be modified: gamma, n_jobs(threads)
    bayes_cv_tuner = BayesSearchCV(
        estimator = xgb.XGBClassifier(            
            use_label_encoder=False,
            n_jobs = -1,
            objective = 'binary:logistic',
            eval_metric = 'aucpr',
            tree_method='approx',
            booster="gbtree"
            ),
        search_spaces = {
            'learning_rate': Real(low=0.01, high=2, prior='log-uniform'),
            'min_child_weight': (0, 10),
            'max_depth': (2, 20),
            'max_delta_step': (0, 20),
            'subsample': (0.01, 1.0, 'uniform'),
            'colsample_bytree': (0.01, 1.0, 'uniform'),
            'colsample_bylevel': (0.01, 1.0, 'uniform'),
            'reg_lambda': (1, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1.0, 'log-uniform'),
            'gamma': (1e-9, 0.5, 'log-uniform'),
            'n_estimators': (50, 100, 150),
            'scale_pos_weight': (1, 500, 'log-uniform')
            },    
        scoring = 'roc_auc',
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
            ),
        n_jobs = -1,
        n_iter = 100,   
        verbose = 0,
        refit = True,
        random_state =42
        )
    
    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""
        
        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
        
        # Get current parameters and the best parameters    
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))
        
        # Save all model results
        clf_name = bayes_cv_tuner.estimator.__class__.__name__
        all_models.to_csv(picpath + "/" + clf_name +"_cv_results.csv")

    result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
    
    #puts out all the estimation parameters
    #test=result.cv_results_
    
    #gets the best estimator from bayesian opt
    model=result.best_estimator_
    #if best estimator not working
    #model= model.fit(X_train, y_train, verbose=True, eval_metric="aucpr")
        
    #The accuracy of the model is calculated and printed
    y_pred = model.predict(X_test) 
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions) 

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    
    """   
    
    #The Probabilities of predicted targets are saved to a table
    proba = model.predict_proba(X_test) 
    mpv, fop=calibration_curve(y_true=y_test, y_prob=proba[:,1], n_bins=10)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    plt.savefig(picpath + '/calimatrix', format = "png")
    plt.clf() 
        
    # plot feature importance
    xgb.plot_importance(model, max_num_features = 20)
    plt.savefig(picpath + '/featurimportanceplot', format = "png")
    plt.clf()
 
    #Confusion plot (makes sense when the value is binary classified)
    conf = plot_confusion_matrix(model,
                          X_test,
                          y_test,
                          display_labels=["Have no stroke", "Have a stroke"])

    plt.savefig(picpath + '/confmatrix', format = "png")
    plt.clf() 

    #Density plot for probality prediction
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

    #Code for printing out the xgb Tree calculated and make it pretty    
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
    """
    return accuracy, model, corrDropCol, X_test, y_test #TODO corrDropCol löschen wenn endpoint fertig ist?
 