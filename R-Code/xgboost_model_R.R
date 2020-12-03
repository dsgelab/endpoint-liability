# This script is for fitting an XGBoost model to the prepared data.

#setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
library(caret)
library(tidyverse)
library(data.table)
library(tictoc)

# Form a grid for the tuning
tunegrid <- expand.grid(
  #nrounds = seq(from = 200, to = nrounds, by = 50),
  nrounds = 100,
  #eta = c(0.025, 0.05, 0.1, 0.3),
  eta = c(0.1),
  #max_depth = c(2, 3, 4, 5, 6),
  max_depth = c(3),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1)


# BELOW NOT NEEDED, xgb automatically has parallel processing
# enable parallel processing
#library(doParallel)
#cl <- makePSOCKcluster(10) #use 10 processors
#registerDoParallel(cl)

set.seed(1010)
tic()
trc <- trainControl(method = "cv", 
                    number = 10, # 10-fold CV
                    verboseIter = FALSE,
                    summaryFunction = twoClassSummary, # needed for ROC
                    classProbs = TRUE) # needed for ROC
xgb_model <- train(I9_STR_EXH ~ ., data = train_data,
                     method = "xgbTree",
                     metric = "ROC",
                     #tuneGrid = tunegrid,
                     tuneLength = 40,
                     trControl = trc)
toc()

#stopCluster(cl)

saveRDS(xgb_model, file="xgb_model.rds")

# confusionMatrix(predictions, variable)