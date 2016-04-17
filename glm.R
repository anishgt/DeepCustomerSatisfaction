
train <- read.csv('/Users/Sreekanth/Spring/BI_algo/Capstone/train.csv',na.strings=c(""))
test <- read.csv('/Users/Sreekanth/Spring/BI_algo/Capstone/test.csv',na.strings=c(""))



# sapply(data,function(x) sum(is.na(x)))
# 
# sapply(data, function(x) length(unique(x)))

dim(train)

#missmap(data, main = "Missing values vs observed")

meanValues <- 0

#length(data[372])

#data[,1]

# for(i in 1:370)
# {
#   #j <- i+1
#     x <- data[,i]
#     meanValues[i] <- mean(x, na.rm = TRUE)
#     for (k in which(is.na(data[,i])))
#     {
#       data[k, i] <- meanValues[i]
#     }
# }

for(colname in names(train))
{
  train[[colname]][is.na(train[[colname]])] <- mean(train[[colname]],na.rm=T)
}


test.id = test$ID
train$ID <- NULL
test$ID <- NULL
train.y <- train$TARGET
train$TARGET <- NULL

# Removing constant values

for(colname in names(train))
{
  if(length(unique(train[[colname]]))==1)
  {
    train[[colname]] <- NULL
    test[[colname]] <- NULL
  }
}

features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
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

## PCA
st.pca <- prcomp(train, center = TRUE, scale. = TRUE)
st.pca.pred <- predict(st.pca, test)

train <- as.data.frame(st.pca$x[,1:100])
test <- as.data.frame(st.pca.pred[,1:100])
#

train$TARGET <- train.y
dim(train)
glmModel <- glm(TARGET ~ .,family=binomial(link='logit'),data=train)

summary(glmModel)

pred <- predict(glmModel,newdata=test,type='response')
pred <- ifelse(pred > 0.5,1,0)

output = data.frame(test.id,pred)
names(output) = c("ID","TARGET")
write.table(output,file = "/Users/Sreekanth/Spring/BI_algo/Capstone/predictions.csv", sep=",",col.names = T, row.names = F)
