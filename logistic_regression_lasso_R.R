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

#data <- fread('endpointsWithDrugs.csv')
tic()
load(file="workspace image.rds")
toc()



data <- endpointsWithDrugs

rm(endpointsWithDrugs)

#specify endpoints to remove manually
exclude_endpoints_manual <- c("I9_STR_SAH",
                              "I9_SEQULAE", # Sequelae of cerebrovascular disease 
                              "I9_STR"
                             )

endpoints_and_drugs <- c(child_endpoints, drug_names)

endpoints_and_drugs <- setdiff(endpoints_and_drugs, exclude_endpoints_manual)

endpoints_and_drugs <- c("I9_STR_EXH", endpoints_and_drugs)
data <- data[,setdiff(names(data), exclude_endpoints_manual)]

#specify number of individuals to take
n_individuals <- nrow(data)

data <- data[sample(1:nrow(data), size=n_individuals, replace=FALSE),]


# remove columns with low number of events
rm_features <- c(NULL)
for (feature in endpoints_and_drugs) {
    if (sum(data[,feature]) < nrow(data)*0.005) {
        rm_features <- c(rm_features, feature)
    }
}
print(paste(length(rm_features), "features removed due to number of events under 1% of data"))

data <- data[,names(data)[!(names(data) %in% rm_features)]]

print(dim(data))

data <- data %>%
  select(-FINNGENID) %>%
  mutate(I9_STR_EXH = as.factor(I9_STR_EXH))

levels(data$I9_STR_EXH) <- c("no", "yes")

# Shuffle the dataset
set.seed(1001)
data <- data[sample(1:nrow(data)), ]

# Select train and test datasets
split <- round(0.7*nrow(data))
train_data <- data[1:split, ]
test_data <- data[(split+1):nrow(data), ]

# grid <- expand.grid(alpha = seq(0, 1, 0.05), lambda = seq(0, 50, 5))
grid2 <- expand.grid(alpha = 1, lambda = 5)



set.seed(1001)
#tic()
#lasso_model <- train(I9_STR_EXH ~ ., data = train_data,
#                     method = "glmnet",
#                     metric = "ROC",
#                     tuneGrid = grid2,
#                     trControl = trainControl(method = "cv",
#                                              number = 2,
#                                              search = "random",
#                                              verboseIter = T,
#                                              classProbs = TRUE))
#toc()

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
                   tuneLength = 1, # number of hyperparameter combinations to try
                   trControl = trc)

toc()

## When you are done:
stopCluster(cl)

#lasso_model


## Train Accuracy
p3 <- predict(lasso_model, type = "prob")
p3 <- ifelse(p3[[2]] >= 0.5, T, F)
table(p3, train_data$I9_STR_EXH)
print(sum(diag(table(p3, train_data$I9_STR_EXH)))/ nrow(train_data))

## Test Accuracy
p4 <- predict(lasso_model, newdata = test_data, type = "prob")
p4 <- ifelse(p4[[2]] >= 0.5, T, F)
table(p4, test_data$I9_STR_EXH)
print(sum(diag(table(p4, test_data$I9_STR_EXH)))/ nrow(test_data))


# Make a confusion matrix of the results
p4 <- as.factor(p4)
levels(p4) <- c(0, 1)
confusionMatrix(p4, test_data$I9_STR_EXH)
