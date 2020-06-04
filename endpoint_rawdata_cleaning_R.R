## endpoint_rawdata_cleaning_R.R
# This script filters the endpoint data. 
# Endpoint columns that have no children are included. Endpoint_age-columns are removed.
# First four PCs are included. This can be modified.
# Running this code for 10 000 rows of endpoint data takes under a minute plus the time to write the output csv file.

# Run first this script and then run the output file with the drug_data_wrangling script!

setwd("C:/Users/timos/OneDrive/Documents/Data-analyysirotaatio/fake_data_analysis")
library(data.table)
library(tidyverse)
library(tictoc)
library(readxl)

# Get the data. Endpoint data and endpoint explanations data.
endpoints <- fread("fake_endpoint_data.csv.gz")
explanations <- read_excel("endpoint_explanations.xlsx")

# Convert NAs to zeros
endpoints[is.na(endpoints)] <- 0

# Filter explanation data so that only rows where include is NA AND level is not 1 or 2.
child_explanations <- explanations %>%
  filter(is.na(INCLUDE)) %>%
  filter(LEVEL!=1 & LEVEL!=2) %>%
  select('NAME')

# Filter the first row which is not any explanation!
child_explanations <- child_explanations[-1,]

parent_explanations <- explanations %>%
  filter(!is.na(INCLUDE)) %>%
  select('NAME', 'INCLUDE', 'LEVEL')

########################
# Dimension reduction part 1

# First arrange children_explanations dataframe so that it shows 
# the parents and parents other children also.
parents <- data.frame('NAME'=child_explanations$NAME, 'PARENT'=NA)
for (i in 1:nrow(child_explanations)) {
   par <- parent_explanations %>%
    filter(grepl(child_explanations$NAME[i], INCLUDE)) %>%
    select('NAME')
   if (nrow(par)>0) { 
     parents[i, 2] <- par
     }
}
child_explanations <- left_join(child_explanations, parents, by='NAME')

# Threshold proportion
prop <- 0.005

child_endpoints <- names(endpoints)[names(endpoints) %in% child_explanations$NAME]

# Form a character vector for parents that are included
included_parents <- character()
for (i in 1:length(child_endpoints)) {
  name <- child_endpoints[i]
  if (sum(endpoints[, ..name])/nrow(endpoints)<prop) {
    included_parents <- c(included_parents, child_explanations[child_explanations$NAME==name, 'PARENT'])
  }
}

included_parents <- unique(included_parents)
print(paste(length(included_parents), "parent endpoints are included"))

# Form a character vector of the children whose parents are included.
removed_children <- character()
for (parent in included_parents) {
  removed_children <- c(removed_children, strsplit(parent_explanations[parent_explanations$NAME==parent, 'INCLUDE'][[1]], "\\|")[[1]])
}
removed_children <- unique(removed_children)
print(paste(length(removed_children), "child endpoints were removed"))


# Now that we know the included parents nad the removed children,
# we form a vector containing all the wanted endpoints.
# To be included, the endpoint need to be IN child_explanations AND NOT IN removed_children,
# OR it has to be IN included_parents.
selected_endpoints <- names(endpoints)[(names(endpoints) %in% child_endpoints) & !(names(endpoints) %in% removed_children)]
selected_endpoints <- c(selected_endpoints, names(endpoints)[names(endpoints) %in% included_parents])
selected_endpoints <- c(selected_endpoints, 'I9_STR_EXH')
selected_endpoints <- unique(selected_endpoints)
selected_endpoints <- sort(selected_endpoints)


# ##################
# # Dimension reduction part 2
# # We calculate polychoric correlations for the selected endpoints
# # and then we can unselect highly correlated ones.
# library(polycor)
# 
# cors <- matrix(data = NA, nrow = length(selected_endpoints), ncol = length(selected_endpoints))
# tic()
# for (i in 5:length(selected_endpoints)) {
#   for (j in 1:length(selected_endpoints)) {
#     i_endpoint <- selected_endpoints[i]
#     j_endpoint <- selected_endpoints[j]
#     cors[i, j] <- polychor(endpoints[, ..i_endpoint][[1]], endpoints[, ..j_endpoint][[1]])
#   }
# }
# toc()
# # 27.61s for 1 row
# # save.image(file="workspace image2.rds", compress=FALSE)
# 
# 
# for (i in 1:ncol(cors)) {
#   cors[i, 1:i] <- NA
# }
# 
# rownames(cors) <- selected_endpoints
# colnames(cors) <- selected_endpoints
# 
# cors_vector <- as.vector(cors)
# cors_vector <- cors_vector[!is.na(cors_vector)]
# 
# # Draw a histogram of the correlations
# hist(cors_vector, main = 'Distribution of variable correlations')
# 
# # Draw a cumulative distribution funtion of the correlations
# plot(ecdf(cors_vector), main ='Cumulative distribution of variable correlations')
# 
# ##########

# Calculate birth years
endpoints <- endpoints %>%
  mutate(BL_YEAR = BL_YEAR - BL_AGE)
names(endpoints)[names(endpoints)=='BL_YEAR'] <- 'BIRTH_YEAR'


# Put the "sex" column to rigth form.
endpoints <- endpoints %>%
  mutate(SEX = ifelse(SEX == 'male', 1, 0))
endpoints$SEX <- as.numeric(endpoints$SEX)
names(endpoints)[names(endpoints)=='SEX'] <- 'SEX_male'


# Select the wanted columns from the data. We want to have birth year, sex and the right endpoints.
# We could include PCs here also by adding them!
endpoints <- endpoints %>%
  select('FINNGENID', 'PC1', 'PC2', 'PC3', 'PC4', 'BIRTH_YEAR', 'FU_END_AGE', 'SEX_male', selected_endpoints)

rm(explanations)
rm(parent_explanations)
rm(child_explanations)
rm(included_parents)
rm(removed_children)
rm(parents)


# Threshold 0.1 -> 634 variables
# Threshold 0.01 -> 642 variables
# Threshold 0.005 -> 658 variables
# Threshold 0.001 -> 787 variables
# Threshold 0.0001 -> 1560 variables