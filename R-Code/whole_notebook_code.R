## get_data_from_finngen_R.R

# get endpoints and purchases from FinnGen data
setwd("/home/jsjukara/")
library(data.table) # for using fread
library(tidyverse)
library(tictoc)

# list columns to keep from phenotype files

keepcols_endbig <- c("FINNGENID","FU_END_AGE", "_AGE",
                     'DEATH','DEATH_AGE','DEATH_YEAR')

# load two phenotype data files
end <- fread("zcat R5_COV_PHENO_V1.txt.gz")

endbig <- fread("finngen_R5_V2_endpoint.txt")
#endbig <- endbig[FINNGENID %in% ids,..keepcols_endbig]

long <- fread("finngen_R5_v2_detailed_longitudinal.txt")

pcs <- paste("PC",1:10, sep="")
keepcols_end <- c("FINNGENID", pcs)
removecols_endbig <- c("SUBSET_COV")
end <- end[,..keepcols_end] # only keep columns of interest, done with data.table

endbig <- data.frame(endbig)
endbig <- endbig[,-c(grep("SUBSET_COV", names(endbig)))]

endpoints <- left_join(end, endbig, by="FINNGENID")

drugPurchases <- long[SOURCE=="PURCH",]
names(drugPurchases)[5:8] <- c("ATC_CODE", "SAIR", "VNRO", "PLKM")
drugPurchases <- as.data.frame(drugPurchases)


rm(long)
rm(end)
rm(endbig)

## endpoint_rawdata_cleaning_R.R
# This script filters the endpoint data. 
# Endpoint columns that have no children are included. Endpoint_age-columns are removed.
# First four PCs are included. This can be modified.
# Running this code for 10 000 rows of endpoint data takes under a minute plus the time to write the output csv file.

# Run first this script and then run the output file with the drug_data_wrangling script!

#setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
library(data.table)
library(tidyverse)
library(tictoc)
library(readxl)

# Get the data. Endpoint data and endpoint explanations data.
#endpoints <- fread("fake_endpoint_data.csv.gz")
explanations <- read_excel("endpoint_explanations.xlsx")

# Filter explanation data so that only rows where include is NA are included.
explanations <- explanations %>%
  filter(is.na(INCLUDE)) %>%
  select('NAME')
# Make a vector of explanation columns
explanations <- c(explanations$NAME)

# Calculate birth years
endpoints <- endpoints %>%
  mutate(BL_YEAR = BL_YEAR - BL_AGE)
names(endpoints)[names(endpoints)=='BL_YEAR'] <- 'BIRTH_YEAR'


# Filter those names of the endpoint data which are the same as in explanation vector.
names <- names(endpoints)
child_endpoints <- character(0)
tic()
for (n in names) {
  for (expl in explanations) {
    if (n==expl) child_endpoints <- c(child_endpoints, n)
  }
}
toc()


endpoints <- endpoints %>%
  mutate(SEX = ifelse(SEX == 'male', 1, 0))
endpoints$SEX <- as.numeric(endpoints$SEX)
names(endpoints)[names(endpoints)=='SEX'] <- 'SEX_male'

# Select the wanted columns from the data. We want to have birth year, sex and the right endpoints.
# We could include PCs here also by adding them!
endpoints <- endpoints %>%
  select('FINNGENID', 'PC1', 'PC2', 'PC3', 'PC4', 'BIRTH_YEAR', 'FU_END_AGE', 'SEX_male', child_endpoints)

# Convert NAs to zeros
endpoints[is.na(endpoints)] <- 0

# fwrite?
#fwrite(endpoints, 'endpoints_cleaned2.csv', row.names = FALSE)

## drug_data_combining_R.R
# In this R scipt drug purchase data is manipulated and connected to the cleaned endpoint data.
# Running this code for 10 000 patients takes about 5 mins. In the end data is saved as csv file to the working directory.
# Run endpoint data with R script endpoint_data_cleaning_R first!!

# Make sure the working directory is right.
#setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")

library(data.table)
library(tidyverse)
library(tictoc)

tic()

# Get the data.
# fread reads the data to datatable form
#endpoints <- fread("endpoints_cleaned2.csv")
#drugPurchases <- fread("fake_purchase_data.csv.gz")

# OPTIONAL!
# Next we take 1000 unique IDs to analysis
# This is due to RStudio performance issues
# Remember to attach tidyverse package

#ids_1000 <- endpoints$FINNGENID[1:1000]
#endpoints <- endpoints %>% 
#  filter(FINNGENID %in% ids_1000)
#drugPurchases <- drugPurchases %>% 
#  filter(FINNGENID %in% ids_1000)

# Next we truncate the ATC-codes to 5 character level
# and filter with that
drugPurchases <- drugPurchases %>%
  mutate(ATC_CODE = substr(ATC_CODE, 1, 5)) %>%
  filter(nchar(ATC_CODE) == 5) %>%
  group_by(ATC_CODE, FINNGENID) %>%
  summarise(all_purchases = sum(PLKM))

# Take up unique atcs and ids
atcs <- unique(drugPurchases$ATC_CODE)
ids <- unique(drugPurchases$FINNGENID)

# Initializing data frame for cumulative purchases
cum_drugPurchases <- data.frame(FINNGENID = ids)


# For each row in drugPurchases we put the number of purchases to our data frame
# It takes 194s to run the code below for 10 000 patients
# The code below is another option of doing this!!!

#for (i in 1:length(drugPurchases$FINNGENID)) {
#  id <- drugPurchases$FINNGENID[i]
#  atc <- drugPurchases$ATC_CODE[i]
#  cum_drugPurchases[cum_drugPurchases$FINNGENID==id, atc] <- drugPurchases$all_purchases[i]
#}


tic()
for (n in 1:length(atcs)) {
  #get data frame of sum_PKOKO for those with at least 1 purchase
  sum_PKOKO <- drugPurchases %>%
    ungroup() %>%
    filter(ATC_CODE == atcs[n]) %>%
    select(FINNGENID, all_purchases)
  names(sum_PKOKO)[2] <- atcs[n]
  
  #join those values with at least 1 purchase (others are left as NA)
  cum_drugPurchases <- left_join(cum_drugPurchases, sum_PKOKO, by = 'FINNGENID')
  
}
#toc()

# Convert NAs to zeros
cum_drugPurchases[is.na(cum_drugPurchases)] <- 0

# Convert drug purchases to binary yes or no (similar to endpoints)
bin_drugPurchases <- cum_drugPurchases
for (i in 2:ncol(bin_drugPurchases)) {
    bin_drugPurchases[,i] <- ifelse(bin_drugPurchases[,i]>1, 1, 0)
}

# And lastly, join drug data to endpoint data
endpointsWithDrugs <- left_join(endpoints, bin_drugPurchases, by = 'FINNGENID')

# impute NA as 0
for (i in 1:ncol(endpointsWithDrugs)) {
    endpointsWithDrugs[,i] <- ifelse(is.na(endpointsWithDrugs[,i]), 0, endpointsWithDrugs[,i])
}


#store drug names
drug_names <- names(cum_drugPurchases)[2:ncol(cum_drugPurchases)]

# remove unneeded files
rm(endpoints)
rm(bin_drugPurchases)
rm(cum_drugPurchases)
rm(drugPurchases)

#fwrite(endpointsWithDrugs, 'endpointsWithDrugs2.csv', row.names = FALSE)
# Save directly as .rds without compression (quicker to write and read, preserves formatting unlike .csv)
save.image(file="workspace image.rds", compress=FALSE)



print("The end!")
toc()

#benchmarking stuff
setwd("/home/jsjukara/")
library(data.table) # for using fread
library(tidyverse)
library(SCCS)
library(lubridate) # for manipulating dates
library(tictoc)
endpointsWithDrugs <- readRDS(file="endpointsWithDrugs.rds")

n_subjects <- nrow(endpointsWithDrugs)
n_features <- 200
keepcols <- names(endpointsWithDrugs)[sample(1:ncol(endpointsWithDrugs), size=n_features, replace=FALSE)]
keepcols <- unique(c("FINNGENID", "I9_STR_EXH", keepcols)) # use unique to delete I9_STR_EXH if it's included twice
data <- endpointsWithDrugs[sample(1:nrow(endpointsWithDrugs), size=n_subjects, replace=FALSE), keepcols]
data <- data %>%
  select(-FINNGENID) %>%
  mutate(I9_STR_EXH = as.factor(data$I9_STR_EXH))



levels(data$I9_STR_EXH) <- c("no", "yes")

rm_cols <- c(NULL)
for (i in 2:ncol(data)) {
    if (sum(data[,i]) == 0) {
        rm_cols <- c(rm_cols, names(data)[i])
    }
}
length(rm_cols)
length(names(data))

data <- data[,names(data)[!(names(data) %in% rm_cols)]]
dim(data)

## Preprocess data for the models
#data <- fread('endpointsWithDrugs.csv')
library(data.table) # for using fread
library(tidyverse)
setwd("/home/jsjukara/")
load(file="workspace image.rds")

data <- endpointsWithDrugs

rm(endpointsWithDrugs)

#specify endpoints to remove manually
exclude_endpoints_manual <- c("I9_STR_SAH",
                              "I9_SEQULAE", # Sequelae of cerebrovascular disease 
                              "I9_STR",
                             "IX_CIRCULATORY")

endpoints_and_drugs <- c(child_endpoints, drug_names)

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
print(paste(length(rm_features), "features removed due to number of events under", paste(inclusion_threshold, "%",sep=""), "of data"))

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

## Elastic net logistic regression

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

#10 fold CV, 20 tunelength,  no parallel, 0.7 split, under 1000 endpoints deleted
45620.857/(60*60)

# all individuals, 10fold CV, 40 tune, 0.7 split, 0.5% threshold for feature inclusion
42734.329/(60*60)/40

time_per_tune <- 2004.28/(60*60)

12/time_per_tune

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

print(object.size(lasso_model), units="Mb")

names(lasso_model)

lasso_model$bestTune

lasso_model$results %>%
    arrange(desc(ROC))

names(lasso_model$finalModel)

betas <- as.data.frame(as.matrix(lasso_model$finalModel$beta))

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


prob_test <- predict(lasso_model, newdata = test_data, type = "prob")$yes
test_I9_STR_EXH <- ifelse(as.integer(test_data$I9_STR_EXH) == 1, 0, 1)
roc_obj <- roc(test_I9_STR_EXH, prob_test)
auc(roc_obj)

plot.roc(test_I9_STR_EXH, prob_test)

temp <- as.matrix(temp)
temp <- as.data.frame(temp)
temp$var <- row.names(temp)
names(temp) <- c("beta", "feature")
head(temp)
temp %>%
    filter(beta > 0.0001) %>%
    arrange(desc(beta)) %>%
    mutate(OR = exp(beta))
test_I9_STR_EXH <- ifelse(as.integer(test_data$I9_STR_EXH) == 1, 0, 1)
roc_obj <- roc(test_I9_STR_EXH, prob_test)
auc(roc_obj)

plot.roc(test_I9_STR_EXH, prob_test)

$$Y = x_1b_1 + x_2b_2 + \varepsilon$$

$$Y = x_1b_1 + x_2b_2 + x_1x_2b_3+ \varepsilon$$

$$Y = x_1b_1 + b_2 + x_1b_3+ \varepsilon$$
$$Y = x_1(b_1 + b_3) + b_2 + \varepsilon$$

temp <- data.frame(I9_STR_EXH = test_I9_STR_EXH)
temp$p4 <- predict(lasso_model, newdata = test_data, type = "prob")

#cal_obj <- calibration(I9_STR_EXH ~ p4,
#                       data = temp,
#                       cuts = 13)
#plot(cal_obj, type = "l", auto.key = list(columns = 3,
#                                          lines = TRUE,
#                                          points = FALSE))

#doesn't work yet

## Random forest

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

#1000 sec
rf_model_save1 <- rf_model

# determine vector of mtry to try
#mtrys <- floor(sqrt(seq(0.2, 2, 0.2)*nrow(data)))

# enable parallel processing
#cl <- makePSOCKcluster(10) #use 10 processors

#trc <- trainControl(method = "cv", 
#                    number = 10, # 10-fold CV
#                   search = "random", # use random search instead of grid search
#                   verboseIter = FALSE,
#                   summaryFunction = twoClassSummary, # needed for ROC
#                   classProbs = TRUE) # needed for ROC

names(rf_model)

prob_test <- predict(rf_model, newdata = test_data, type = "prob")$yes
test_I9_STR_EXH <- ifelse(as.integer(test_data$I9_STR_EXH) == 1, 0, 1)
roc_obj <- roc(test_I9_STR_EXH, prob_test)
auc(roc_obj)

plot.roc(test_I9_STR_EXH, prob_test)

varImp(rf_model, scale = TRUE)

## XGBoost

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

6/(626/(60*60))

#/10 data -> 59sec
rf_xgb_save1 <- xgb_model
 