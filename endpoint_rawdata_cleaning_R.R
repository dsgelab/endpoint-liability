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

# Below is another option for doing column selection.
#names <- names(endpoints)
#child_endpoints <- character(0)
#tic()
#for (n in names) {
#  for (expl in explanations) {
#    if (grepl(expl, n)) child_endpoints <- c(child_endpoints, n)
#  }
#}
#toc()

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
fwrite(endpoints, 'endpoints_cleaned2.csv', row.names = FALSE)

print("The end!")