library(kknn)
library(MASS)
library(caret)

download.file("https://raw.githubusercontent.com/ChicagoBoothML/HelpR/master/docv.R", "docv.R")
source("docv.R")

#--------------------------------------------------
#cv version for polynomial regression
docvpoly = function(x,y,d,nfold=10,doran=TRUE,verbose=TRUE) {
  return(docv(x,y,matrix(d,ncol=1),dopoly,mse,nfold=nfold,doran=doran,verbose=verbose))
}

dopoly=function(x,y,xp,d) {
  train = data.frame(x,y=y)
  test = data.frame(xp); names(test) = names(train)[1:(ncol(train)-1)]
  fit = lm(y~poly(x, degree=d[1]), train)
  return (predict.lm(fit, test))
}
#--------------------------------------------------

carData <- read.csv("https://raw.githubusercontent.com/ChicagoBoothML/DATA___UsedCars/master/UsedCars.csv",
                    header= TRUE)

##Split Data into Training set and test set
numRowsFull <- nrow(carData)
ntrain <- floor(0.75*numRowsFull) ## Number of observations for training data 
set.seed(99) #set seed for random split so it is reproducible

trData <- sample(1:numRowsFull,ntrain) #Split the data randomly
train <- carData[trData,] #training data
test <- carData[-trData,] #testdata

##Polynomial Fit of the Data

priceTraining <- train$price
mileageTraining <- train$mileage
numFolds <- c(15,10,5)
polyDegree <- c(1:12)
cv <- list()

for (n in 1:length(numFolds)) {
  folds <- createFolds(priceTraining, k = numFolds[n],list = TRUE, returnTrain = FALSE)
  cvmean=rep(0,length(polyDegree)) #store results for mean
  for (j in folds){
    testIndex <- j
    trainingTemp <- train[-testIndex,]
    testTemp <-  train[testIndex,]
    for (i in polyDegree){
      model <- lm(price ~ poly(mileage,i), data=trainingTemp)
      model_prediction <- predict(model,testTemp)
      residualFold <- (testTemp$price-model_prediction)^2
      msError <- sum(residualFold)
      cvmean[i] <- cvmean[i] + msError 
    }
  }
  cv[n] <- list(sqrt(cvmean/length(priceTraining)))
}

rgy <- range(c(cv[[1]],cv[[2]],cv[[3]]))
plot(polyDegree,cv[[1]],type="l",col="red", main = "RMSE vs Polynomial Degree for each CV",ylim = rgy, lwd=2,
     cex.lab=0.8, xlab="Polynomial Degree", ylab="RMSE")
lines(polyDegree,cv[[2]],col="blue",lwd=2)
lines(polyDegree,cv[[3]],col="green",lwd=2)
legend("topleft",legend=c("10-Fold","5-fold","15 fold"),
       col=c("blue","green","red"),lwd=2,cex=0.8)

cvFinal <- (cv[[1]]+cv[[2]]+cv[[3]])/3
plot(polyDegree, cvFinal, type="l",col="red", main = "Average RMSE vs Polynomial Degree Over All CVs",xlab="Polynomial Degree", ylab="RMSE")

degreeBest <- polyDegree[which.min(cvFinal)]

cat('\n')
cat('Optimal Degree is:',degreeBest, '\n')

###Retraining Using the Entire Training Set for Optimal Polynomial Degree

model <- lm(price ~ poly(mileage,5), data=train)
print(summary(model))
cat('Correlation between residuals and fitted values:', cor(model$fitted.values,model$residuals),'\n')

###Plot the Data Along with the Model Fit

train <- train[order(train$mileage),] ##Sort the dataset by mileage so we can plot polynomial fit using lines without issue
test <- test[order(test$mileage),]

plot(train$mileage,ttrain <- train[order(train$mileage),]$price,main = "Scatterplot with Polynomial Fit from Training Data", xlab="Mileage",ylab="Price")
lines(train$mileage,predict(model,train), col = "red", lwd = 2)


