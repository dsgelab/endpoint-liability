## get_data_from_finngen_R.R

# get endpoints and purchases from FinnGen data
setwd("/home/jsjukara/")
library(data.table) # for using fread
library(tidyverse)
library(SCCS)
library(lubridate) # for manipulating dates
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