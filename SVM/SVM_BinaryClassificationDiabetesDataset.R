diabetes.data<- read.csv(file="C:/Users/sathv/OneDrive/Stats Research Project/diabetes_prediction_dataset.csv", 
header=TRUE, sep=",")

diabetes.data$smoking_history<- ifelse(diabetes.data$smoking_history %in% c("current", "former"),1,0)  
diabetes.data$gender<- ifelse(diabetes.data$gender=='Male',1,0)
                                       

#SPLITTING DATA INTO 80% TRAINING AND 20% TESTING SETS 
set.seed(966452)
sample <- sample(c(TRUE, FALSE), nrow(diabetes.data), replace=TRUE, prob=c(0.8,0.2))
train<- diabetes.data[sample,]
test<- diabetes.data[!sample,]

train.x<- data.matrix(train[-9])
train.y<- data.matrix(train[9])
test.x<- data.matrix(test[-9])
test.y<- data.matrix(test[9])

install.packages("e1071")
library(e1071)

#FITTING SVM WITH LINEAR KERNEL
svm.class<- svm(as.factor(diabetes) ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, 
data=train, kernel="linear")

#computing prediction accuracy for testing data
pred.y<-as.numeric(predict(svm.class, test.x))-1
match <- numeric(length(pred.y))

for (i in 1:length(pred.y))
  match[i]<- ifelse(test.y[i]==pred.y[i], 1,0)
print(paste("accuracy=", round(mean(match), digits=4)))

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
print("Linear Kernel Confusion Matrix Results:")
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
svm.class <- svm(as.factor(diabetes) ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, data=train, kernel="linear", probability=TRUE)

svm.prob <- predict(svm.class, newdata=test, probability=TRUE)
prob_attr <- attr(svm.prob, "probabilities")
test$prob_1 <- prob_attr[, "1"]

tpos<- matrix(NA, nrow=nrow(test), ncol=102)
fpos<- matrix(NA, nrow=nrow(test), ncol=102)
tneg<- matrix(NA, nrow=nrow(test), ncol=102)
fneg<- matrix(NA, nrow=nrow(test), ncol=102)


for (i in 0:101) {
  tpos[,i+1]<- ifelse(test$diabetes==1 & test$prob_1>=0.01*i,1,0)
  fpos[,i+1]<- ifelse(test$diabetes==0 & test$prob_1>=0.01*i, 1,0)
  tneg[,i+1]<- ifelse(test$diabetes==0 & test$prob_1<0.01*i,1,0)
  fneg[,i+1]<- ifelse(test$diabetes==1 & test$prob_1<0.01*i,1,0)
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


#FITTING SVM WITH POLYNOMIAL KERNEL
svm.class<- svm(as.factor(diabetes) ~  gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, 
data=train, kernel="polynomial")

#computing prediction accuracy for testing data
pred.y<- as.numeric(predict(svm.class, test.x))-1

for (i in 1:length(pred.y))
  match[i]<- ifelse(test.y[i]==pred.y[i], 1,0)
print(paste("accuracy=", round(mean(match), digits=4)))

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
print("Polynomial Kernel Confusion Matrix Results:")
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
svm.class <- svm(as.factor(diabetes) ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, data=train, kernel="polynomial", probability=TRUE)

svm.prob <- predict(svm.class, newdata=test, probability=TRUE)
prob_attr <- attr(svm.prob, "probabilities")
test$prob_1 <- prob_attr[, "1"]

tpos<- matrix(NA, nrow=nrow(test), ncol=102)
fpos<- matrix(NA, nrow=nrow(test), ncol=102)
tneg<- matrix(NA, nrow=nrow(test), ncol=102)
fneg<- matrix(NA, nrow=nrow(test), ncol=102)


for (i in 0:101) {
  tpos[,i+1]<- ifelse(test$diabetes==1 & test$prob_1>=0.01*i,1,0)
  fpos[,i+1]<- ifelse(test$diabetes==0 & test$prob_1>=0.01*i, 1,0)
  tneg[,i+1]<- ifelse(test$diabetes==0 & test$prob_1<0.01*i,1,0)
  fneg[,i+1]<- ifelse(test$diabetes==1 & test$prob_1<0.01*i,1,0)
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


#FITTING SVM WITH RADIAL KERNEL
svm.class<- svm(as.factor(diabetes) ~  gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, 
data=train, kernel="radial")

#computing prediction accuracy for testing data
pred.y<- as.numeric(predict(svm.class, test.x))-1

for (i in 1:length(pred.y))
  match[i]<- ifelse(test.y[i]==pred.y[i], 1,0)
print(paste("accuracy=", round(mean(match), digits=4)))


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
print("Radial Kernel Confusion Matrix Results:")
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
svm.class <- svm(as.factor(diabetes) ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, data=train, kernel="radial", probability=TRUE)


svm.prob <- predict(svm.class, newdata=test, probability=TRUE)
prob_attr <- attr(svm.prob, "probabilities")
test$prob_1 <- prob_attr[, "1"]

tpos<- matrix(NA, nrow=nrow(test), ncol=102)
fpos<- matrix(NA, nrow=nrow(test), ncol=102)
tneg<- matrix(NA, nrow=nrow(test), ncol=102)
fneg<- matrix(NA, nrow=nrow(test), ncol=102)


for (i in 0:101) {
  tpos[,i+1]<- ifelse(test$diabetes==1 & test$prob_1>=0.01*i,1,0)
  fpos[,i+1]<- ifelse(test$diabetes==0 & test$prob_1>=0.01*i, 1,0)
  tneg[,i+1]<- ifelse(test$diabetes==0 & test$prob_1<0.01*i,1,0)
  fneg[,i+1]<- ifelse(test$diabetes==1 & test$prob_1<0.01*i,1,0)
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


#FITTING SVM WITH SIGMOID KERNEL
svm.class<- svm(as.factor(diabetes) ~  gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, 
data=train, kernel="sigmoid")

#computing prediction accuracy for testing data
pred.y<- as.numeric(predict(svm.class, test.x))-1

for (i in 1:length(pred.y))
  match[i]<- ifelse(test.y[i]==pred.y[i], 1,0)
print(paste("accuracy=", round(mean(match), digits=4)))


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
print("Sigmoid Kernel Confusion Matrix Results:")
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
svm.class <- svm(as.factor(diabetes) ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, data=train, kernel="sigmoid", probability=TRUE)

svm.prob <- predict(svm.class, newdata=test, probability=TRUE)
prob_attr <- attr(svm.prob, "probabilities")
test$prob_1 <- prob_attr[, "1"]

tpos<- matrix(NA, nrow=nrow(test), ncol=102)
fpos<- matrix(NA, nrow=nrow(test), ncol=102)
tneg<- matrix(NA, nrow=nrow(test), ncol=102)
fneg<- matrix(NA, nrow=nrow(test), ncol=102)


for (i in 0:101) {
  tpos[,i+1]<- ifelse(test$diabetes==1 & test$prob_1>=0.01*i,1,0)
  fpos[,i+1]<- ifelse(test$diabetes==0 & test$prob_1>=0.01*i, 1,0)
  tneg[,i+1]<- ifelse(test$diabetes==0 & test$prob_1<0.01*i,1,0)
  fneg[,i+1]<- ifelse(test$diabetes==1 & test$prob_1<0.01*i,1,0)
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

