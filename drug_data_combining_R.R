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