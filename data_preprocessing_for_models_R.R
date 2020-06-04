## Preprocess data for the models
#data <- fread('endpointsWithDrugs.csv')
library(data.table) # for using fread
library(tidyverse)
#setwd("/home/jsjukara/")
setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
load(file="workspace image.rds")

data <- endpointsWithDrugs

rm(endpointsWithDrugs)

#specify endpoints to remove manually
exclude_endpoints_manual <- c("I9_STR_SAH",
                              "I9_SEQULAE", # Sequelae of cerebrovascular disease 
                              "I9_STR",
                              "IX_CIRCULATORY")

endpoints_and_drugs <- c(selected_endpoints, drug_names)

endpoints_and_drugs <- setdiff(endpoints_and_drugs, exclude_endpoints_manual)

endpoints_and_drugs <- c("I9_STR_EXH", endpoints_and_drugs)
data <- data[,setdiff(names(data), exclude_endpoints_manual)]

#specify number of individuals to take
n_individuals <- nrow(data)

data <- data[sample(1:nrow(data), size=n_individuals, replace=FALSE),]

# remove columns with low number of events
inclusion_threshold <- 0.005
rm_features <- c(NULL)
for (feature in endpoints_and_drugs) {
  if (sum(data[,feature]) < nrow(data)*inclusion_threshold) {
    rm_features <- c(rm_features, feature)
  }
}
print(paste(length(rm_features), "features removed due to number of events under", paste(inclusion_threshold*100, "%",sep=""), "of data"))

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

fix(train_data)
rm_features
selected_endpoints
