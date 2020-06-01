## logistic_regression_lasso_R.R

# This script is for fitting logistic regression model with elastic net penalty to the prepared data.

#setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
setwd("/home/jsjukara/")
library(caret)
library(tidyverse)
library(data.table)
library(e1071)
library(tictoc)
library(pROC)

#grid2 <- expand.grid(alpha = 1, lambda = 5)

set.seed(1001)

# enable parallel processing
library(doParallel)
cl <- makePSOCKcluster(10) #use 10 processors
registerDoParallel(cl)


tic()
trc <- trainControl(method = "cv", 
                    number = 10, # 10-fold CV
                    search = "random", # use random search instead of grid search
                    verboseIter = FALSE,
                    summaryFunction = twoClassSummary, # needed for ROC
                    classProbs = TRUE) # needed for ROC

lasso_model <- train(I9_STR_EXH ~ ., data = train_data, 
                   method = "glmnet", 
                   metric = "ROC",
                   tuneLength = 40, # number of hyperparameter combinations to try
                   trControl = trc)

toc()

stopCluster(cl)

saveRDS(lasso_model, file="lasso_model.rds")