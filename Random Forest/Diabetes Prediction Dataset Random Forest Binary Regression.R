diabetes.data<- read.csv(file="C:/Users/sathv/OneDrive/Stats Research Project/diabetes_prediction_dataset.csv", 
header=TRUE, sep=",")

#SPLITTING DATA INTO SUBSETS
class0 <- subset(diabetes.data, diabetes == 0)
class1 <- subset(diabetes.data, diabetes == 1)

#CALCULATING HOW MANY 0'S NEEDED FOR ALL 1'S TO MAKE A GIVEN RATIO
n1 <- nrow(class1)
n0_needed <- 10*nrow(class1) #USED TO TEST IF LESS 0'S INCREASE ACCURACY (did not affect) 

print(n0_needed)

#SAMPLING THE 0'S
set.seed(447558)
class0_sample <- class0[sample(nrow(class0), size = n0_needed), ]

print(nrow(class0_sample))
print(nrow(class1))
print(nrow(class0))
#ADDING THE SAMPLE TO THE 1'S
ratio_data = rbind(class1,class0_sample)

#SPLITTING DATA INTO 80% TRAINING AND 20% TESTING SETS 
set.seed(447558)

n_train0 <- floor(0.8 * nrow(class0_sample))
train0<- class0_sample[sample(nrow(class0_sample), size = n_train0),]
test0<- class0_sample[!(rownames(class0_sample) %in% rownames(train0)),]

n_train1 <- floor(0.8 * nrow(class1))
train1<- class1[sample(nrow(class1), size = n_train1),]
test1<- class1[!(rownames(class1) %in% rownames(train1)),]

train <- rbind(train0, train1)
test <- rbind(test1, test0)
#BUILDING RANDOM FOREST BINARY CLASSIFIER
library(randomForest)
rf.class<- randomForest(as.factor(diabetes) ~ gender + age + hypertension	+ heart_disease + bmi + HbA1c_level + blood_glucose_level + smoking_history,
data=train, ntree=300, mtry=5, maxnodes=30)

#DISPLAYING FEATURE IMPORTANCE
print(importance(rf.class,type=2)) 

#COMPUTING PREDICTION ACCURACY FOR TESTING DATA 
predclass<- predict(rf.class, newdata=test)
test<- cbind(test,predclass)

accuracy<- c()
n<- nrow(test)
for (i in 1:n)
  accuracy[i]<- ifelse(test$diabetes[i]==test$predclass[i],1,0)

print(accuracy<- mean(accuracy))

#calculating confusion matrix
tp <- sum(predclass == 1 & test$diabetes == 1)
fp <- sum(predclass == 1 & test$diabetes == 0)
tn <- sum(predclass == 0 & test$diabetes == 0)
fn <- sum(predclass == 0 & test$diabetes == 1)
total <- length(test$diabetes)

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
print("Random Forest Confusion Matrix Results:")
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
prob_rf <- predict(rf.class, newdata=test, type="prob")
test$prob_1 <- prob_rf[, "1"]

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

