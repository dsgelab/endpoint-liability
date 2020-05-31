# This script is for fitting an XGBoost model to the prepared data.

setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
library(caret)
library(tidyverse)
library(data.table)

data <- fread('endpointsWithDrugs.csv')

data <- data %>%
  select(-FINNGENID)

# Shuffle the dataset
set.seed(1010)
data <- data[sample(1:nrow(data)), ]

# Select train and test datasets
split <- round(0.8*length(data$BIRTH_YEAR))
train_data <- data[1:split, ]
test_data <- data[(split+1):length(data$BIRTH_YEAR), ]

# Form a grid for the tuning
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1)


set.seed(1010)
xgboost_tree <- train(I9_STR_EXH ~ ., data = train_data,
                     method = "xgbTree",
                     metric = "ROC",
                     tuneGrid = tune_grid,
                     trControl = trainControl(method = "cv",
                                              number = 10))




# confusionMatrix(predictions, variable)
