library(RSNNS)
library(caret)
rm(list=ls(all=T))
setwd("/media/anishgt/D_Drive/githububuntu/DeepCustomerSatisfaction")
train <- read.csv('santander/train.csv',na.strings=c(""))
test <- read.csv('santander/test.csv',na.strings=c(""))

test.id = test$ID
train$ID <- NULL
test$ID <- NULL
train.y <- train$TARGET
train$TARGET <- NULL
toRemove <- c()
for(colname in names(train))
{
  if(length(unique(train[[colname]]))==1)
  {
    toRemove <- c(toRemove, colname)
  }
}

features_pair <- combn(names(train), 2, simplify = F)

for(pair in features_pair) 
{
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}
feature.names <- setdiff(names(train), toRemove)

dim(train)
dim(test)

train <- train[, feature.names]
test <- test[, feature.names]
train.decoded=decodeClassLabels(train.y)
save(train, test, train.y, test.id, feature.names,file='anish_preprocessed.RData')

model <- mlp(train, y=train.y, size=c(100,10,10), maxit = 2000, initFunc = "Randomize_Weights", initFuncParams = c(-0.8, 0.8), learnFunc = "Std_Backpropagation", learnFuncParams = c(0.01,0), linOut=FALSE)
#, learnFuncParams =0.1,maxit = 100
predictresults= predict(model, dataset.test[,-1])
predictions=apply(predictresults,1,function(x) if (x[1]>=x[2]) return(3) else return(8) )
print(caret::confusionMatrix(predictions,dataset.test$V1))
