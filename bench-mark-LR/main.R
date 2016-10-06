# this script read the dt from the train (in order to improve time train.truncated 
# is the first 0.1 million of train), split the dt into Train and Test

rm(list=ls())
library(data.table)
library(caret)#partition
library(glmnet)
library(Matrix)
library(pROC)
library(ROCR)
library(plyr)
library(stringr)
library(lubridate)
library(ggplot2)


dt <- data.table(read.csv(file="../data/train",head=TRUE,sep=","))
dt[is.na(dt)] <- 'na'

# remove the column 'id' from predictor because id is unique to each row
# remove the column 'click'
x <- names(dt)[-c(1,2)] 
target <- 'click'
fm <- as.formula(paste(target ,"~",
                       paste(x,collapse=" + "),
                       sep=" "))
cat('The fomula of the model is: \n')
print(fm)

# change all the column for modelling to factor 
for (i in x) {
  dt[[i]] <- as.factor(dt[[i]]) # convert the class to factor
}
# ======================== randomly data partition ============================
dt$index<-seq(nrow(dt))
set.seed(123)
TrainIndex <- createDataPartition(y = unique(dt$index), p=0.9, list=FALSE)
dt$partition <- ifelse(dt$index %in% (dt$index)[TrainIndex],'train','test')
cat("label count of train data: \n")
print(table(dt[partition == 'train'][[target]]))
cat("label count of test data: \n")
print(table(dt[partition == 'test'][[target]]))
dt$index <- NULL

Train <- dt[partition == 'train']
Test <- dt[partition == 'test']

# Train$partition <- NULL
# Test$partition <- NULL
# write.table(Train, "train.log.txt", row.names = FALSE, quote = FALSE, sep="\t")
# write.table(Test, "test.log.txt", row.names = FALSE, quote = FALSE, sep="\t")
# =====================bench mark logistic regression ==========================
# ------------------index features according to the Train ----------------------
maxindex <- 0
# add intercept 
featindex <- data.table(name_value = c("intercept"),
                        index = c(1))
maxindex <- maxindex + 1
# index the features from the Train
for (i in seq_len(length(x))) {
  # get the unique value of the selected col, always add 'na'
  col.selected.feat <- unique(rbindlist(list(list('na'),Train[, x[i], with = FALSE]))) 
  setnames(col.selected.feat,names(col.selected.feat),"column")
  feat.name_value = paste(x[i],col.selected.feat$column, sep = ":") # colname:colvalue 
  featindex.tmp <- data.table(name_value = feat.name_value,
                              index = seq(length(feat.name_value)) + maxindex)
  featindex <- list(featindex, featindex.tmp)
  featindex <- rbindlist(featindex, use.names = TRUE, fill = TRUE)
  maxindex <- maxindex + length(feat.name_value)
}

rm(featindex.tmp)

featindex$name_value <- as.character(featindex$name_value) # colname:colvalue
featindex$index <- as.numeric(featindex$index) # index

# convert featindex into a hash table
hash <- function(x) {
  # create a new environment
  e <- new.env(hash = TRUE, 
               size = nrow(x),       
               parent = emptyenv()) 
  
  apply(x, 1, function(col) assign(col[1],     
                                   as.numeric(col[2]), 
                                   envir = e))
  return(e)
}  

featindex_hash <- hash(featindex) # featindex$'colname:colvalue' = index

# ------------------------------index Train------------------------------------
start_time<-Sys.time()
sink("train.log.txt")
for (row in seq_len(nrow(Train))) {
  # for (row in 1:10) {
  cat(Train[row, 'click', with = FALSE]$click)
  cat('\t1:1')
  for (i in seq_len(length(x))) {
    feat.name_value = paste(x[i],":",Train[row, x[i], with = FALSE][[x[i]]], sep = "")
    cat(paste('\t', featindex_hash[[feat.name_value]],':1', sep = ""))
  }
  cat('\n')
}
sink()s
cat ('time usage for indexing train:',
     round(difftime(Sys.time(),start_time,units = "mins"),3),'mins\n')
# ------------------------------index Test-------------------------------------
start_time<-Sys.time()
sink('test.log.txt')
for (row in seq_len(nrow(Test))) {
  # for (row in 1:10) {
  cat(Test[row,'click', with = FALSE]$click)
  cat('\t1:1')
  for (i in seq_len(length(x))) {
    feat.name_value = paste(x[i],":",Test[row, x[i], with = FALSE][[x[i]]], sep = "")
    if (!exists(feat.name_value, envir = featindex_hash)) {
      feat.name_value = paste(x[i],':na', sep = "")
    }
    cat(paste('\t', featindex_hash[[feat.name_value]],':1', sep = ""))
  }
  cat('\n') 
}
sink()
cat ('time usage for indexing test:',
     round(difftime(Sys.time(),start_time,units = "mins"),3),'mins\n')
# -----------------------------start training----------------------------------
source('./bench_mark_LR.R')
