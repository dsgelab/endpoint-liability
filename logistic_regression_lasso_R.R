# This script is for fitting logistic regression model with elastic net penalty to the prepared data.

setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
library(caret)
library(tidyverse)
library(data.table)
library(e1071)
library(tictoc)

data <- fread('endpointsWithDrugs.csv')

data <- data %>%
  select(-FINNGENID) %>%
  mutate(I9_STR_EXH = as.factor(data$I9_STR_EXH))

# Shuffle the dataset
set.seed(1001)
data <- data[sample(1:nrow(data)), ]

# Select train and test datasets
split <- round(0.8*length(data$BIRTH_YEAR))
train_data <- data[1:split, ]
test_data <- data[(split+1):length(data$BIRTH_YEAR), ]

# grid <- expand.grid(alpha = seq(0, 1, 0.05), lambda = seq(0, 50, 5))
grid2 <- expand.grid(alpha = 1, lambda = 5)



set.seed(1001)
tic()
lasso_model <- train(I9_STR_EXH ~ ., data = train_data,
                     method = "glmnet",
                     metric = "ROC",
                     tuneGrid = grid2,
                     trControl = trainControl(method = "cv",
                                              number = 2,
                                              search = "random",
                                              verboseIter = T,
                                              classProbs = TRUE))
toc()
# 817s!!!

lasso_model


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


