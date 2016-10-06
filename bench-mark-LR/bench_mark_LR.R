# bench mark logistic regression
# requires file 'train.log.txt' and 'test.log.txt'
library(pROC)

start_time<-Sys.time()
#initialize
bufferCaseNum <- 100000
eta <- 0.01
lamb <- 1E-6
trainRounds <- 100
initWeight <- 0.05
featWeight <- new.env(hash = TRUE, 
                      parent = emptyenv()) 

nextInitWeight <- function (initWeight) {
  weight <- runif(1,-0.5,0.5) * initWeight
  return(weight)
}

ints <- function (s) {
  # convert the list of character to list of int
  res = c()
  for (i in seq_len(length(s))) {
    res[length(res) + 1] <- strtoi(s[i])
  }
  return(res)
}

sigmoid <- function (p) {
  res <- 1.0 / (1.0 + exp(-1 * p))
  return(res)
}


trainFile <- 'train.log.txt'
testFile <- 'test.log.txt'
sink('training.output.txt')
# training the logistic model
for (rnd in 1:trainRounds) {
  fi <- file(trainFile,open="r")
  linn <-readLines(fi)
  lineNum <- 0
  trainData <- list()
  for (i in 1:length(linn)){
    # cat(linn[i])
    # cat('\n')
    trainData[[length(trainData) +1]] <- ints(strsplit(gsub(":1", "", linn[i]), '\t')[[1]])
    lineNum <- (lineNum + 1) %% bufferCaseNum
    if (lineNum == 0) {
      for (j in 1:length(trainData)){
        data <- trainData[[j]]
        clk <- data[1]
        pred <- 0.0
        for (k in 2:length(data)){
          feat = data[k]
          if (!(exists(toString(feat), envir = featWeight))) {
            assign(toString(feat),     
                   nextInitWeight(initWeight), 
                   envir = featWeight)
          }
          pred <- pred + featWeight[[toString(feat)]]
        }
        pred <- sigmoid(pred)
        print(pred)
        for (k in 2:length(data)){
          feat = data[k]
          featWeight[[toString(feat)]] <- featWeight[[toString(feat)]] * (1-lamb) + eta * (clk - pred)
        }
      }
      trainData <- list()
    }
    
  }
  if (length(trainData) > 0) {
    for (j in 1:length(trainData)) {
      data <- trainData[[j]]
      clk <- data[1]
      pred <- 0.0
      for (k in 2:length(data)) {
        feat <- data[k]
        if(!(exists(toString(feat), envir = featWeight))) {
          assign(toString(feat),
                 nextInitWeight(initWeight),
                 envir = featWeight)
        }
        pred <- pred + featWeight[[toString(feat)]]
      }
      pred <- sigmoid(pred)
      for (k in 2:length(data)) {
        feat <- data[k]
        featWeight[[toString(feat)]] <- featWeight[[toString(feat)]] * (1-lamb) + eta * (clk - pred)
      }
    }
  }
  
  
  close(fi)
  # test for this round
  y <- c()
  yp <- c()
  fi <- file(testFile,open="r")
  linn <-readLines(fi)
  for (i in 1:length(linn)){
    data <- ints(strsplit(gsub(":1", "", linn[i]), '\t')[[1]])
    clk <- data[1]
    pred <- 0.0
    for (k in 2:length(data)) {
      feat <- data[k]
      if(exists(toString(feat), envir = featWeight)) {
        # if seen_feature
        pred <- pred + featWeight[[toString(feat)]]
      }
    }
    pred <- sigmoid(pred)
    y[length(y) + 1] <- clk
    yp[length(yp) +1] <- pred
  }
  close(fi)
  auc <- gsub("[^0-9.]", "", auc(y, yp))
  rmse <- sqrt(mean((y-yp)^2))
  cat(paste(toString(rnd),'\t',toString(auc),'\t',toString(rmse),'\n', sep = ""))
}
cat ('time usage of bench-mark logistic regression: ',round(difftime(Sys.time(),start_time,units = "mins"),3),'mins\n')
sink()
save(featWeight, file = 'featWeight.Rda')

# output the feature weight


