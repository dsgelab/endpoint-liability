# This script is for fitting random forests model to the prepared data.

#setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
setwd("/home/jsjukara/")
library(caret)
library(tidyverse)
library(data.table)
library(tictoc)
library(randomForest)
library(pROC)

# determine vector of mtry to try
mtrys <- floor(sqrt(seq(0.4, 2.4, 0.2)*nrow(data)))
print(paste("mtry's to try:" mtrys))

tunegrid <- expand.grid(.mtry = mtrys)

# enable parallel processing
library(doParallel)
cl <- makePSOCKcluster(10) #use 10 processors
registerDoParallel(cl)

set.seed(1009)
tic()
trc <- trainControl(method = "cv", 
                    number = 10, # 10-fold CV
                    verboseIter = FALSE,
                    summaryFunction = twoClassSummary, # needed for ROC
                    classProbs = TRUE) # needed for ROC

rf_model <- train(I9_STR_EXH ~ ., data = train_data, 
                   method = "rf", 
                   metric = "ROC",
                   tuneGrid=tunegrid,
                   trControl = trc)
toc()

stopCluster(cl)

saveRDS(rf_model, file="rf_model.rds")

# confusionMatrix(predictions, variable)