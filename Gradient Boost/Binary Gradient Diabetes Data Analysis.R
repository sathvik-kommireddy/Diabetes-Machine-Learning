install.packages("xgboost")
library(xgboost)

diabetes.data<- read.csv(file="C:/Users/sathv/OneDrive/Stats Research Project/diabetes_prediction_dataset.csv", 
header=TRUE, sep=",")

#SPLITTING DATA INTO 80% TRAINING AND 20% TESTING SETS 
set.seed(447558)
sample <- sample(c(TRUE, FALSE), nrow(diabetes.data), 
replace=TRUE, prob=c(0.8,0.2))
train<- diabetes.data[sample,]
test<- diabetes.data[!sample,]

train.x<- data.matrix(train[-9])
train.y<- data.matrix(train[9])
test.x<- data.matrix(test[-9])
test.y<- data.matrix(test[9])

#FITTING GRADIENT BOOSTED BINARY CLASSIFIER
xgb.class<- xgboost(data=train.x, label=train.y, 
max.depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.5, 
nrounds=1000, objective="binary:logistic")

#DISPLAYING FEATURE IMPORTANCE
print(xgb.importance(colnames(train.x), model=xgb.class))

#COMPUTING PREDICTION ACCURACY FOR TESTING DATA 
pred.prob<- predict(xgb.class, test.x)

len<- length(pred.prob)
pred.diabetes<- c()
match<- c()
for (i in 1:len){
  pred.diabetes[i]<- ifelse(pred.prob[i]>=0.5, 1,0)
  match[i]<- ifelse(test.y[i]==pred.diabetes[i], 1,0)
}
print(prop<- sum(match)/len)

#calculating confusion matrix
tp <- sum(pred.diabetes == 1 & test.y == 1)
fp <- sum(pred.diabetes == 1 & test.y == 0)
tn <- sum(pred.diabetes == 0 & test.y == 0)
fn <- sum(pred.diabetes == 0 & test.y == 1)
total <- length(test.y)

# Metrics
accuracy <- (tp + tn) / total
misclassrate <- (fp + fn) / total
sensitivity <- tp / (tp + fn)
FNR <- fn / (tp + fn)
specificity <- tn / (fp + tn)
FPR <- fp / (fp + tn)
precision <- tp / (tp + fp)
NPV <- tn / (fn + tn)
F1score <- 2 * tp / (2 * tp + fn + fp)

# Print results
print("XGBoost Confusion Matrix Results:")
print(paste("TP =", tp, " FP =", fp, " TN =", tn, " FN =", fn, "Total =", total))
print(paste("Accuracy =", round(accuracy, 4)))
print(paste("MisclassRate =", round(misclassrate, 4)))
print(paste("Sensitivity =", round(sensitivity, 4)))
print(paste("FNR =", round(FNR, 4)))
print(paste("Specificity =", round(specificity, 4)))
print(paste("FPR =", round(FPR, 4)))
print(paste("Precision =", round(precision, 4)))
print(paste("NPV =", round(NPV, 4)))
print(paste("F1 Score =", round(F1score, 4)))
####ROC CURVE CODE

#COMPUTING CONFUSION MATRICES AND PERFORMANCE MEASURES FOR TESTING SET
#FOR A RANGE OF CUT-OFFS

tpos<- matrix(NA, nrow=nrow(test), ncol=102)
fpos<- matrix(NA, nrow=nrow(test), ncol=102)
tneg<- matrix(NA, nrow=nrow(test), ncol=102)
fneg<- matrix(NA, nrow=nrow(test), ncol=102)


for (i in 0:101) {
  tpos[,i+1]<- ifelse(test.y==1 & pred.prob>=0.01*i,1,0)
  fpos[,i+1]<- ifelse(test.y==0 & pred.prob>=0.01*i, 1,0)
  tneg[,i+1]<- ifelse(test.y==0 & pred.prob<0.01*i,1,0)
  fneg[,i+1]<- ifelse(test.y==1 & pred.prob<0.01*i,1,0)
}

tp<- c()
fp<- c()
tn<- c()
fn<- c()
accuracy<- c()
misclassrate<- c()
sensitivity<- c()
specificity<- c()
oneminusspec<- c()
cutoff<- c()


for (i in 1:102) {
  tp[i]<- sum(tpos[,i])
  fp[i]<- sum(fpos[,i])
  tn[i]<- sum(tneg[,i])
  fn[i]<- sum(fneg[,i])
  total<- nrow(test)
  accuracy[i]<- (tp[i]+tn[i])/total
  misclassrate[i]<- (fp[i]+fn[i])/total
  sensitivity[i]<- tp[i]/(tp[i]+fn[i])
  specificity[i]<- tn[i]/(fp[i]+tn[i])
  oneminusspec[i]<- fp[i]/(fp[i]+tn[i])
  cutoff[i]<- 0.01*(i-1)
}

#PLOTTING ROC CURVE
plot(oneminusspec, sensitivity, type="l", lty=1, main="The Receiver 
Operating Characteristic Curve", xlab="1-Specificity", ylab="Sensitivity")
points(oneminusspec, sensitivity, pch=0) #pch=plot character, 0=square

#REPORTING MEASURES FOR THE POINT ON ROC CURVE CLOSEST TO THE IDEAL POINT (0,1)
distance<- c()
for (i in 1:102)
  distance[i]<- sqrt(oneminusspec[i]^2+(1-sensitivity[i])^2)

measures<- cbind(accuracy, misclassrate, sensitivity, specificity, distance, cutoff, tp, fp, tn, fn)
min.dist<- min(distance)
print(measures)

#COMPUTING AREA UNDER THE ROC CURVE
sensitivity<- sensitivity[order(sensitivity)]
oneminusspec<- oneminusspec[order(oneminusspec)]

library(Hmisc) #Harrell Miscellaneous packages
lagx<- Lag(oneminusspec,shift=1)
lagy<- Lag(sensitivity, shift=1)
lagx[is.na(lagx)]<- 0
lagy[is.na(lagy)]<- 0
trapezoid<- (oneminusspec-lagx)*(sensitivity+lagy)/2
print(AUC<- sum(trapezoid))



