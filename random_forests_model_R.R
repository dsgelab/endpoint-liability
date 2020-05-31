# This script is for fitting random forests model to the prepared data.

setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
library(caret)
library(tidyverse)
library(data.table)
library(tictoc)

data <- fread('endpointsWithDrugs.csv')

data <- data %>%
  select(-FINNGENID) %>%
  mutate(I9_STR_EXH = as.factor(data$I9_STR_EXH))

# Shuffle the dataset
set.seed(1009)
data <- data[sample(1:nrow(data)), ]

# Select train and test datasets
split <- round(0.8*length(data$BIRTH_YEAR))
train_data <- data[1:split, ]
test_data <- data[(split+1):length(data$BIRTH_YEAR), ]

# grid <- expand.grid(mtry = 1:4, ntree = c(200, 300, 400),
#                    nodesize = 1:10)
grid2 <- expand.grid(.mtry = 2)


# Only mtry can be tuned with caret!! 
set.seed(1009)
tic()
random_forests <- train(I9_STR_EXH ~ ., data = train_data,
                     method = "rf",
                     metric = "ROC",
                     tuneGrid = grid2,
                     trControl = trainControl(method = "cv",
                                              number = 10,
                                              search = "random",
                                              verboseIter = T))
toc()




# confusionMatrix(predictions, variable)
