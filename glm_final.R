library(glmnet)
library(caret)
library(pROC)

preprocess <- function(dataset ){
	
	#test$ID <- NULL

	# Removing constant values
	toRemove <- c()
	for(colname in names(dataset))
	{
	  if(length(unique(dataset[[colname]]))==1)
	  {
		toRemove <- c(toRemove, colname)
		#dataset[[colname]] <- NULL
		#test[[colname]] <- NULL
	  }
	}

	features_pair <- combn(names(dataset), 2, simplify = F)
	
	for(pair in features_pair) 
	{
	  f1 <- pair[1]
	  f2 <- pair[2]

	  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
		if (all(dataset[[f1]] == dataset[[f2]])) {
		  cat(f1, "and", f2, "are equals.\n")
		  toRemove <- c(toRemove, f2)
		}
	  }
	}
	#feature.names <- setdiff(names(dataset), toRemove)

	dim(dataset)
	#dim(test)
	#dataset <- dataset[, feature.names]
	return(toRemove)
}


doPCA <- function(train, test ){
	#test <- test[, feature.names]
	st.pca <- prcomp(train, center = TRUE, scale. = TRUE)
	st.pca.pred <- predict(st.pca, test)

	train <- as.data.frame(st.pca$x[,1:10])
	test <- as.data.frame(st.pca.pred[,1:10])
	
	return(list(train,test))
	}

predGLM <- function(trainloc, thresh, testloc, test.id){
	 
	subtrain0=trainloc[sample(which(trainloc$TARGET == 0),5000),]
	subtrain1=trainloc[which(trainloc$TARGET==1),]
	#subtrain1=train[sample(which(train.y == 1),1000),]
	subtrain=rbind(subtrain0,subtrain1)

	subvalid0=trainloc[sample(which(trainloc$TARGET == 0),1000),]
	subvalid1=trainloc[sample(which(trainloc$TARGET == 1),1000),]
	subvalid=rbind(subvalid0,subvalid1)
	
	#pca.d <- doPCA(subtrain, subvalid)
	
	#subtrain <- pca.d[[1]]
	#subvalid <- pca.d[[2]]
	glmModel <- glm(TARGET ~ .,family=binomial(link='logit'),data=subtrain)
	pred <- predict(glmModel,newdata=subvalid,type='response')
	hist(pred)
	pred <- ifelse(pred > thresh,1,0)
	print(caret::confusionMatrix(pred,subvalid$TARGET))
	roc(pred, subvalid$TARGET)	
	
	pred <- predict(glmModel,newdata=testloc,type='response')
	pred.n <- ifelse(pred > thresh,1,0)
	output = data.frame(test.id,pred.n)
	names(output) = c("ID","TARGET")
	write.table(output,file = "glmpredictions.csv", sep=",",col.names = T, row.names = F)
}

predGLMLasso <- function(train, thresh, test, test.id){
	subtrain0=train[sample(which(train$TARGET == 0),5000),]
	subtrain1=train[which(train$TARGET==1),]
	#subtrain1=train[sample(which(train.y == 1),1000),]
	subtrain=rbind(subtrain0,subtrain1)

	subvalid0=train[sample(which(train$TARGET == 0),1000),]
	subvalid1=train[sample(which(train$TARGET == 1),1000),]
	subvalid=rbind(subvalid0,subvalid1)
	X=as.matrix(subtrain[,1:(ncol(subtrain)-1)])
	y=subtrain$TARGET
	#newdata=data.frame( x=X, y=y)
	#bestAIC= bestglm(newdata, family=binomial, IC="BIC", nvmax=10)


	cv <- cv.glmnet(X,y)
	#model=glmnet(X,y,family = "binomial",lambda=cv$lambda.min)
	result <- glmnet(X, as.factor(y), family = "binomial", alpha=1)
	pred=predict(result, as.matrix(subvalid[,1:(ncol(subvalid)-1)]), type="response", s=cv$lambda.min)
	hist(pred)
	pred.n <- ifelse(pred > thresh,1,0)
	print(confusionMatrix(pred.n, as.factor(subvalid[,ncol(subvalid)])))
	roc(as.vector(pred.n), as.vector((subvalid[,ncol(subvalid)])))
	
	pred=predict(result, as.matrix(test[,1:ncol(test)]), type="response", s=cv$lambda.min)
	pred.n <- ifelse(pred > thresh,1,0)
	output = data.frame(test.id,pred.n)
	names(output) = c("ID","TARGET")
	write.table(output,file = "lassopredictions.csv", sep=",",col.names = T, row.names = F)
}


train <- read.csv('santander/train.csv',na.strings=c(""))
test <- read.csv('santander/test.csv',na.strings=c(""))

train.id = train$ID
train.y <- train$TARGET

test.id = test$ID
		
train$TARGET <- NULL
train$ID <- NULL

test$ID <- NULL

toRemove=preprocess(train)
feature.names <- setdiff(names(train), toRemove)
train <- train[, feature.names]
test <- test[, feature.names]

#preprocessed.test = preprocess(test, FALSE)
#test <- preprocessed.test[[1]]
pca.d <- doPCA(train, test)
pca.train <- pca.d[[1]]
pca.test <- pca.d[[2]]


train$TARGET <- train.y
pca.train$TARGET = train.y

predGLM(train=pca.train,thresh= 0.2, pca.test, test.id)

predGLMLasso(train=train, thresh=0.3, test, test.id)




			
		
