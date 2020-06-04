## logistic_regression_lasso_R.R

# This script is for fitting logistic regression model with elastic net penalty to the prepared data.

setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
#setwd("/home/jsjukara/")
library(caret)
library(tidyverse)
library(data.table)
library(e1071)
library(tictoc)
library(pROC)



set.seed(1001)

# enable parallel processing
library(doParallel)
cl <- makePSOCKcluster(3) #use 10 processors
registerDoParallel(cl)


tic()
trc <- trainControl(method = "cv", 
                    number = 10, # 10-fold CV
                    search = "random", # use random search instead of grid search
                    verboseIter = FALSE,
                    summaryFunction = twoClassSummary, # needed for ROC
                    classProbs = TRUE) # needed for ROC

lasso_model <- train(I9_STR_EXH ~ ., data = train_70, 
                     method = "glmnet", 
                     metric = "ROC",
                     tuneLength = 1, # number of hyperparameter combinations to try
                     trControl = trc)

toc()

################
# Here we test the time complexity of the model with different n and p.

p <- seq(100, 900, 100)

df <- data.frame(p=p, runtime=NA)


for (i in 1:length(p)) {
  p_data <- train_data[1:1000, c(1:7, 180, sample(c(8:179, 181:929), p[i]))]
  runtimes <- rep(NA, 10)
  for (j in 1:10) {
    tic()
    train(I9_STR_EXH ~ ., data = p_data, 
          method = "glmnet", 
          metric = "ROC",
          tuneLength = 1, # number of hyperparameter combinations to try
          trControl = trc)
    x <- toc()
    runtimes[j] <- x$toc - x$tic
  }
  df$runtime[i] <-  mean(runtimes)
}

plot(df$p, df$runtime, main = 'Runtime ~ number of variables')

# Then test the same again, but this time with different n:s

n <- seq(1000, 5000, 1000)
df2 <- data.frame(n=n, runtime=NA)


for (i in 1:length(n)) {
  p_data <- train_data[1:n[i], ]
  runtimes <- rep(NA, 10)
  for (j in 1:10) {
    tic()
    train(I9_STR_EXH ~ ., data = p_data, 
          method = "glmnet", 
          metric = "ROC",
          tuneLength = 1, # number of hyperparameter combinations to try
          trControl = trc)
    x <- toc()
    runtimes[j] <- x$toc - x$tic
  }
  df2$runtime[i] <-  mean(runtimes)
}

plot(df2$n, df2$runtime, main = 'Runtime ~ number of samples')
################

stopCluster(cl)

saveRDS(lasso_model, file="lasso_model.rds")
