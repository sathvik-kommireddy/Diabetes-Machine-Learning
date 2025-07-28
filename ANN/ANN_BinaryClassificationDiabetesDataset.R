diabetes.data<- read.csv(file="C:/Users/sathv/OneDrive/Stats Research Project/diabetes_prediction_dataset.csv", 
header=TRUE, sep=",")

diabetes.cat<- ifelse(diabetes.data$diabetes==1,1,0)
diabetes.data$smoking_history<- ifelse(diabetes.data$smoking_history %in% c("current", "former"),1,0)  
diabetes.data$gender<- ifelse(diabetes.data$gender=='Male',1,0)           

class0 <- subset(diabetes.data, diabetes == 0)
class1 <- subset(diabetes.data, diabetes == 1)

set.seed(42)
class0_sample <- class0[sample(nrow(class0), size = 4 * nrow(class1)), ]
balanced_data <- rbind(class0_sample, class1)

library(caret)

set.seed(503548)
# Stratified partitioning: ensures train/test have similar class balance
split_index <- createDataPartition(balanced_data$diabetes, p=0.8, list=FALSE)
train <- balanced_data[split_index, ]
test <- balanced_data[-split_index, ]

train$diabetes <- as.numeric(train$diabetes)
test$diabetes <- as.numeric(test$diabetes)

train.x<- data.matrix(train[-9])
train.y<- data.matrix(train[9])
test.x<- data.matrix(test[-9])
test.y<- data.matrix(test[9])

library(neuralnet)

#FITTING ANN WITH LOGISTIC ACTIVATION FUNCTION AND ONE LAYER WITH THREE NEURONS
ann.log.class<- neuralnet(diabetes ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, 
                                   data=train, hidden = 3, act.fct="logistic", stepmax=1e7)

#PLOTTING THE DIAGRAM
plot(ann.log.class)

#COMPUTING PREDICTION ACCURACY FOR TESTING DATA
pred.prob<- predict(ann.log.class, test.x)[,1]

pred.y<- c()
match<- c()
for (i in 1:length(test.y)){
  pred.y[i]<- ifelse(pred.prob[i]>0.5,1,0)
  match[i]<- ifelse(test.y[i]==pred.y[i],1,0)
}

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
print("ANN Confusion Matrix Results:")
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


####################################################################
#FITTING ANN WITH LOGISTIC ACTIVATION FUNCTION AND C(2,3) LAYERS
ann.log23.class<- neuralnet(diabetes ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, 
data=train, hidden=c(2,3), act.fct="logistic", stepmax=1e7)

#PLOTTING THE DIAGRAM
plot(ann.log23.class)

#COMPUTING PREDICTION ACCURACY FOR TESTING DATA
pred.prob<- predict(ann.log23.class, test.x)[,1]

match<- c()
pred.y<- c()
for (i in 1:length(test.y)){
  pred.y[i]<- ifelse(pred.prob[i]>0.5,1,0)
  match[i]<- ifelse(test.y[i]==pred.y[i],1,0)
}

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
print("ANN Confusion Matrix Results:")
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



####################################################################
# Scale the input features (excluding the target variable)
scaled_train_x <- as.data.frame(scale(train[, -9]))  # exclude 'diabetes' column
scaled_test_x <- as.data.frame(scale(test[, -9]))

# Add back the diabetes column
scaled_train <- cbind(scaled_train_x, diabetes = train$diabetes)
scaled_test <- cbind(scaled_test_x, diabetes = test$diabetes)

# Update data matrices for prediction
train.x <- data.matrix(scaled_train[, -9])
train.y <- data.matrix(scaled_train[, 9])
test.x <- data.matrix(scaled_test[, -9])
test.y <- data.matrix(scaled_test[, 9])

#FITTING ANN WITH TANH ACTIVATION FUNCTION
ann.tanh.class<- neuralnet(diabetes ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level, 
data=train, hidden=2, act.fct="tanh", stepmax=1e7)

#PLOTTING THE DIAGRAM
plot(ann.tanh.class)

#COMPUTING PREDICTION ACCURACY FOR TESTING DATA
pred.prob<- predict(ann.tanh.class, test.x)[,1]

match<- c()
pred.y<- c()
for (i in 1:length(test.y)){
  pred.y[i]<- ifelse(pred.prob[i]>0.5,1,0)
  match[i]<- ifelse(test.y[i]==pred.y[i],1,0)
}

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
print("ANN Confusion Matrix Results:")
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
pred.prob <- (pred.prob + 1) / 2

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


