library(Amelia)
data <- read.csv('/home/arjun/capstone/DeepCustomerSatisfaction/santander/train.csv',na.strings=c(""))

sapply(data,function(x) sum(is.na(x)))

sapply(data, function(x) length(unique(x)))

#missmap(data, main = "Missing values vs observed")

meanValues <- 0

#length(data[372])

data[,1]

for(i in 1:370)
{
  #j <- i+1
    x <- data[,j]
    meanValues[i] <- mean(x, na.rm = TRUE)
    for (k in which(is.na(data[,j])))
    {
      data[k, j] <- meanValues[i]
    }
}

#PCA Analysis
prcomp(data, scale = FALSE)

glmModel <- glm(TARGET ~ ., data=data, family="binomial")


summary(glmModel)
