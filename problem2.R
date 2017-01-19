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
numFolds <- 15
#set.seed(118)
folds <- createFolds(priceTraining, k = numFolds,list = TRUE, returnTrain = FALSE)
polyDegree <- c(1:12)

cvmean=rep(0,length(polyDegree)) #store results for mean


##Cross Validation (15 fold)
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

cv1 <- sqrt(cvmean/length(priceTraining))

##Cross Validation (10 Fold)
numFolds <- 10
#set.seed(99)
folds <- createFolds(priceTraining, k = numFolds,list = TRUE, returnTrain = FALSE)

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

cv2 <- sqrt(cvmean/length(priceTraining))

##Cross Validation (5 Fold)
numFolds <- 5
#set.seed(103)
folds <- createFolds(priceTraining, k = numFolds,list = TRUE, returnTrain = FALSE)

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

cv3 <- sqrt(cvmean/length(priceTraining))

rgy <- range(c(cv1,cv2,cv3))
plot(polyDegree,cv1,type="l",col="red", main = "RMSE vs Polynomial Degree for each CV",ylim = rgy, lwd=2,
     cex.lab=0.8, xlab="Polynomial Degree", ylab="RMSE")
lines(polyDegree,cv2,col="blue",lwd=2)
lines(polyDegree,cv3,col="green",lwd=2)
legend("topleft",legend=c("10-Fold","5-fold","15 fold"),
       col=c("blue","green","red"),lwd=2,cex=0.8)

cv <- (cv1+cv2+cv3)/3
plot(polyDegree, cv, type="l",col="red", main = "Average RMSE vs Polynomial Degree Over All CVs",xlab="Polynomial Degree", ylab="RMSE")

degreeBest <- polyDegree[which.min(cv)]

###Sanity Check: Polynomial Cross Validation using Mladen's Package

#set.seed(118)
polyDegree <- 1:12

cvB1 <- docvpoly(matrix(mileageTraining, ncol = 1),priceTraining, polyDegree, nfold = 15)
cvB2 <- docvpoly(matrix(mileageTraining, ncol = 1),priceTraining, polyDegree, nfold = 10)
cvB3 <- docvpoly(matrix(mileageTraining, ncol = 1),priceTraining, polyDegree, nfold = 5)

### Changing to RMSE format

cvB1 <- sqrt(cvB1/length(priceTraining))
cvB2 <- sqrt(cvB2/length(priceTraining))
cvB3 <- sqrt(cvB3/length(priceTraining))

cvB <- (cvB1+cvB2+cvB3)/3

degreeBestCheck <- polyDegree[which.min(cvB)]

cat('\n')
cat('Optimal Degree is:',degreeBest, '\n')
cat('According to Mladen\'s function:',degreeBestCheck,'\n', '\n')

###Retraining Using the Entire Training Set for Optimal Polynomial Degree

model <- lm(price ~ poly(mileage,degreeBest), data=train)
print(summary(model))
cat('Correlation between residuals and fitted values:', cor(model$fitted.values,model$residuals),'\n')

###Plot the Data Along with the Model Fit

train <- train[order(train$mileage),] ##Sort the dataset by mileage so we can plot polynomial fit using lines without issue
test <- test[order(test$mileage),]

plot(train$mileage,ttrain <- train[order(train$mileage),]$price,main = "Scatterplot with Polynomial Fit from Training Data", xlab="Mileage",ylab="Price")
lines(train$mileage,predict(model,train), col = "red", lwd = 2)

mlTest <- kknn(price~mileage, train, train, k = 480, kernel = 'rectangular')

lines(train$mileage, mlTest$fitted.values, col = "blue", lwd = 2)

