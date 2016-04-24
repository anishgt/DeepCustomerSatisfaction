library(glmnet)
train <- read.csv('santander/train.csv',na.strings=c(""))
test <- read.csv('santander/test.csv',na.strings=c(""))

test.id = test$ID
train.y <- train$TARGET
train$TARGET <- NULL
train$ID <- NULL
test$ID <- NULL

subtrain0=train[sample(which(train.y == 0),5000),]
subtrain1=train[which(train.y==1),]
#subtrain1=train[sample(which(train.y == 1),5000),]
subtrain=rbind(subtrain0,subtrain1)
subtrain.y=subtrain$TARGET

subvalid0=train[sample(which(train.y == 0),1000),]
subvalid1=train[sample(which(train.y == 1),1000),]
subvalid=rbind(subvalid0,subvalid1)
subvalid.y=subvalid$TARGET

lasso.names=c("var3","var15","imp_op_var39_comer_ult3","imp_op_var40_comer_ult1","imp_op_var40_efect_ult1","imp_op_var41_comer_ult1","imp_op_var41_ult1","ind_var8_0","ind_var12_0","ind_var13_largo_0","ind_var13","ind_var20_0","ind_var24","ind_var26_cte","ind_var30_0","ind_var30","ind_var31_0","ind_var32_0","ind_var32","ind_var33_0","ind_var39_0","num_var12","num_var14_0","num_var20_0","num_var24","num_op_var40_hace2","num_op_var41_hace2","num_var37_med_ult2","num_var42","saldo_var5","saldo_var8","saldo_var26","saldo_var30","saldo_var37","var36","delta_imp_aport_var13_1y3","delta_imp_compra_var44_1y3","delta_imp_reemb_var17_1y3","delta_num_compra_var44_1y3","delta_num_reemb_var17_1y3","imp_compra_var44_hace3","imp_reemb_var13_ult1","ind_var43_emit_ult1","ind_var43_recib_ult1","var21","num_aport_var13_hace3","num_ent_var16_ult1","num_var22_hace2","num_var22_ult1","num_var22_ult3","num_meses_var5_ult3","num_meses_var8_ult3","num_meses_var13_largo_ult3","num_meses_var17_ult3","num_meses_var39_vig_ult3","num_op_var39_comer_ult1","num_op_var39_efect_ult3","num_reemb_var17_ult1","num_sal_var16_ult1","num_var43_emit_ult1","num_var43_recib_ult1","num_var45_hace3","num_var45_ult1","saldo_medio_var5_hace2","saldo_medio_var5_ult1","saldo_medio_var5_ult3","saldo_medio_var8_hace2","saldo_medio_var8_ult1","saldo_medio_var12_ult3","saldo_medio_var17_hace2","var38")
train <- train[, lasso.names]
test <- test[, lasso.names]
X=as.matrix(subtrain[,2:(ncol(subtrain)-1)])
y=subtrain.y
newdata=data.frame( x=X, y=y)
bestAIC= bestglm(newdata, family=binomial, IC="BIC", nvmax=10)


cv <- cv.glmnet(X,y)
#model=glmnet(X,y,family = "binomial",lambda=cv$lambda.min)
result <- glmnet(X, as.factor(y), family = "binomial", alpha=1)
pred=predict(result, as.matrix(subvalid[,2:(ncol(subvalid)-1)]), type="response", s=cv$lambda.min)
hist(pred)
pred.n <- ifelse(pred > 0.3,1,0)
confusionMatrix(pred.n, as.factor(subvalid[,ncol(subvalid)]))
roc(as.vector(pred.n), subvalid.y)

pred=predict(result, as.matrix(test[,2:ncol(test)]), type="response", s=cv$lambda.min)
pred.n <- ifelse(pred > 0.3,1,0)
output = data.frame(test.id,pred.n)
names(output) = c("ID","TARGET")
write.table(output,file = "lassopredictions.csv", sep=",",col.names = T, row.names = F)

#glmbest features : var3 + imp_op_var41_efect_ult3 + imp_op_var39_ult1 + ind_var13_largo_0 + ind_var30_0 + ind_var32 + num_var24 + saldo_var8 + delta_imp_aport_var13_1y3 + imp_var43_emit_ult1 + num_ent_var16_ult1 + num_meses_var5_ult3 + num_sal_var16_ult1 + num_var45_ult3 + var38 + var15 + imp_op_var41_ult1 + imp_sal_var16_ult1 + ind_var13 + ind_var30 + num_var8_0 + num_op_var41_hace2 + saldo_var26 + delta_imp_venta_var44_1y3 + ind_var43_emit_ult1 + num_var22_hace2 + num_meses_var12_ult3 + num_var43_emit_ult1 + saldo_medio_var5_hace2 + imp_ent_var16_ult1 + imp_op_var39_efect_ult1 + ind_var8_0 + ind_var20_0 + ind_var31_0 + num_var12 + num_var42 + saldo_var30 + delta_num_venta_var44_1y3 + ind_var43_recib_ult1 + num_var22_ult1 + num_meses_var17_ult3 + num_var43_recib_ult1 + saldo_medio_var8_hace2 + imp_op_var39_comer_ult3   + imp_op_var39_efect_ult3   + ind_var12_0               + ind_var24                 + ind_var32_0               + num_var20_0               + saldo_var5                + var36                     + imp_reemb_var17_ult1      + num_aport_var13_hace3     + num_var22_ult3            + num_op_var39_efect_ult3   + num_var45_ult1            + saldo_medio_var12_hace3
names(subtrain[1+which(coef(result,s=cv$lambda.min)[2:length(coef(result,s=cv$lambda.min))]!=0)])
glmmodelafterglmbest = glm(TARGET ~ var3 + imp_op_var41_efect_ult3 + imp_op_var39_ult1 + ind_var13_largo_0 + ind_var30_0 + ind_var32 + num_var24 + saldo_var8 + delta_imp_aport_var13_1y3 + imp_var43_emit_ult1 + num_ent_var16_ult1 + num_meses_var5_ult3 + num_sal_var16_ult1 + num_var45_ult3 + var38 + var15 + imp_op_var41_ult1 + imp_sal_var16_ult1 + ind_var13 + ind_var30 + num_var8_0 + num_op_var41_hace2 + saldo_var26 + delta_imp_venta_var44_1y3 + ind_var43_emit_ult1 + num_var22_hace2 + num_meses_var12_ult3 + num_var43_emit_ult1 + saldo_medio_var5_hace2 + imp_ent_var16_ult1 + imp_op_var39_efect_ult1 + ind_var8_0 + ind_var20_0 + ind_var31_0 + num_var12 + num_var42 + saldo_var30 + delta_num_venta_var44_1y3 + ind_var43_recib_ult1 + num_var22_ult1 + num_meses_var17_ult3 + num_var43_recib_ult1 + saldo_medio_var8_hace2 + imp_op_var39_comer_ult3   + imp_op_var39_efect_ult3   + ind_var12_0               + ind_var24                 + ind_var32_0               + num_var20_0               + saldo_var5                + var36                     + imp_reemb_var17_ult1      + num_aport_var13_hace3     + num_var22_ult3            + num_op_var39_efect_ult3   + num_var45_ult1            + saldo_medio_var12_hace3 ,  data=subtrain, family="binomial")


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

train <- as.data.frame(st.pca$x[,1:10])
test <- as.data.frame(st.pca.pred[,1:10])
#

train$TARGET <- train.y
dim(train)


bestglm.features=c("var15","imp_op_var39_comer_ult3","imp_op_var41_ult1","ind_var13","ind_var30_0","delta_imp_aport_var13_1y3","num_var22_ult1","num_meses_var5_ult3","var38","ind_var30","num_op_var41_hace2","num_var37_med_ult2","saldo_var5","saldo_var37","ind_var43_recib_ult1")
train <- train[, final.features]
test <- test[, final.features]

subtrain0=train[sample(which(train.y == 0),5000),]
subtrain1=train[which(train.y==1),]
subtrain1=train[sample(which(train.y == 1),1000),]
subtrain=rbind(subtrain0,subtrain1)

glmModel <- glm(TARGET ~ .,family=binomial(link='logit'),data=subtrain)
glmModel <- glm(TARGET ~ .,family=binomial(link='logit'),data=subtrain)

summary(glmModel)

subvalid0=train[sample(which(train.y == 0),000),]
subvalid1=train[sample(which(train.y == 1),3000),]
subvalid=rbind(subvalid0,subvalid1)


pred <- predict(glmModel,newdata=subvalid,type='response')
hist(pred)
pred <- ifelse(pred > 0.2,1,0)

print(caret::confusionMatrix(pred.n,subvalid$TARGET))
roc(pred, subvalid$TARGET)

pred <- predict(glmModel,newdata=test,type='response')
pred.n <- ifelse(pred > 0.5,1,0)
output = data.frame(test.id,pred.n)
names(output) = c("ID","TARGET")
write.table(output,file = "glmpredictions.csv", sep=",",col.names = T, row.names = F)
