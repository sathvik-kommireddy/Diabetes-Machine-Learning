diabetes.data<- read.csv(file="C:/Users/sathv/OneDrive/Stats Research Project/diabetes_prediction_dataset.csv", 
header=TRUE, sep=",")

#SPLITTING DATA INTO 80% TRAINING AND 20% TESTING SETS 
library(caTools)
set.seed(704467) 
sample<- sample.split(diabetes.data$diabetes, SplitRatio=0.80)
train<- subset(diabetes.data, sample==TRUE)
test<-  subset(diabetes.data, sample==FALSE)

train.x<- data.matrix(train[-9])
train.y<- data.matrix(train[9])
test.x<- data.matrix(test[-9])
test.y<- data.matrix(test[9])

#TRAINING K-NEAREST NEIGHBOR BINARY CLASSIFIER
library(caret)#classification and regression training
print(train(as.factor(diabetes)~., data=train, method="knn"))

#FITTING OPTIMAL KNN BINARY CLASSIFIER (k=9)
knn.class<- knnreg(train.x, train.y, k=9)

#COMPUTING PREDICTION ACCURACY FOR TESTING DATA 
pred.prob<- predict(knn.class, test.x)

len<- length(pred.prob)
pred.y<- c()
match<- c()
for (i in 1:len){
  pred.y[i]<- ifelse(pred.prob[i]>=0.5, 1,0)
  match[i]<- ifelse(test.y[i]==pred.y[i], 1,0)
}
print(paste("accuracy=",round(mean(match),digits=4)))

#alternative (frugal) way
pred.y1<- floor(0.5+predict(knn.class, test.x))
print(paste("accuracy=", round(1-mean(test.y!=pred.y1),digits=4)))

#calculating confusion matrix
tp <- sum(pred.y == 1 & test.y == 1)
fp <- sum(pred.y == 1 & test.y == 0)
tn <- sum(pred.y == 0 & test.y == 0)
fn <- sum(pred.y == 0 & test.y == 1)
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
print("KNN Confusion Matrix Results:")
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
  tpos[,i+1]<- ifelse(test$diabetes==1 & pred.prob>=0.01*i,1,0)
  fpos[,i+1]<- ifelse(test$diabetes==0 & pred.prob>=0.01*i, 1,0)
  tneg[,i+1]<- ifelse(test$diabetes==0 & pred.prob<0.01*i,1,0)
  fneg[,i+1]<- ifelse(test$diabetes==1 & pred.prob<0.01*i,1,0)
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


