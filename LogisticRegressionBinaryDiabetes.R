diabetes.data<- read.csv(file="C:/Users/sathv/OneDrive/Stats Research Project/diabetes_prediction_dataset.csv", 
header=TRUE, sep=",")
diabetes.data <- subset(diabetes.data, gender != "Other")

# Set reference levels
diabetes.data$gender <- relevel(as.factor(diabetes.data$gender), ref = "Male")
diabetes.data$smoking_history <- relevel(as.factor(diabetes.data$smoking_history), ref = "never")

# Fit logistic model
fitted.model <- glm(diabetes ~ gender + smoking_history + age + hypertension + bmi + HbA1c_level + blood_glucose_level, 
                    data = diabetes.data, family = binomial(link = "logit"))
summary(fitted.model)

#extracting AICC and BIC for fitted model
p <- length(coef(fitted.model))  # total number of model parameters
n <- nrow(diabetes.data)
AICC <- -2 * logLik(fitted.model) + 2 * p * n / (n - p - 1)
print(AICC)
BIC(fitted.model)

#checking model fit
null.model<- glm(diabetes.data$diabetes ~ 1, family=binomial(link=logit))
print(deviance<- -2*(logLik(null.model)-logLik(fitted.model)))
print(p.value<- pchisq(deviance, df=3, lower.tail=FALSE))

#using fitted model for prediction
predict(fitted.model, type="response", 
        newdata = data.frame(gender = "Female", 
                             smoking_history = "current",
                             age = 45, 
                             hypertension = 1, 
                             bmi = 27.5, 
                             HbA1c_level = 6.1, 
                             blood_glucose_level = 140))

# Load packages
library(ggplot2)

# Build coefficient summary from your logistic regression model
model_summary <- summary(fitted.model)

logit_df <- data.frame(
  Feature = rownames(coef(model_summary)),
  Estimate = coef(model_summary)[, "Estimate"],
  Pvalue = coef(model_summary)[, "Pr(>|z|)"]
)

# Remove intercept
logit_df <- subset(logit_df, Feature != "(Intercept)")

# Compute -log10(p-value), cap at 16 (max precision from R)
logit_df$logP <- -log10(logit_df$Pvalue)
logit_df$logP[is.infinite(logit_df$logP)] <- 16

# Plot with correct fill
ggplot(logit_df, aes(x = reorder(Feature, abs(Estimate)), y = Estimate, fill = logP)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(
    low = "lightgray", 
    high = "darkred", 
    name = "-log10(p-value)",
    limits = c(0, 16)
  ) +
  labs(
    title = "Logistic Regression Coefficients with Significance",
    x = "Feature",
    y = "Coefficient (Slope)"
  ) +
  theme_minimal(base_size = 14)
