diabetes.data<- read.csv(file="C:/Users/sathv/OneDrive/Stats Research Project/diabetes_prediction_dataset.csv", 
                         header=TRUE, sep=",")
diabetes.data <- subset(diabetes.data, gender != "Other")
# Set reference levels
diabetes.data$gender <- relevel(as.factor(diabetes.data$gender), ref = "Male")
diabetes.data$smoking_history <- relevel(as.factor(diabetes.data$smoking_history), ref = "never")

#fitting probit model
fitted.model <- glm(diabetes ~ gender + smoking_history + age + hypertension + bmi + HbA1c_level + blood_glucose_level, 
                    data = diabetes.data, family = binomial(link = "probit"))
summary(fitted.model)

#extracting AICC and BIC for fitted model
p <- length(coef(fitted.model))  # total number of model parameters
n <- nrow(diabetes.data)
AICC <- -2 * logLik(fitted.model) + 2 * p * n / (n - p - 1)
print(AICC)
BIC(fitted.model)

#checking model fit
null.model<- glm(diabetes.data$diabetes ~ 1, family=binomial(link=probit))
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


# Load ggplot2
library(ggplot2)

# Step 1: Extract coefficient summary
probit_summary <- summary(fitted.model)

# Step 2: Create data frame with coefficients and p-values
probit_df <- data.frame(
  Feature = rownames(coef(probit_summary)),
  Estimate = coef(probit_summary)[, "Estimate"],
  Pvalue = coef(probit_summary)[, "Pr(>|z|)"]
)

# Step 3: Remove intercept if you don't want it plotted
probit_df <- subset(probit_df, Feature != "(Intercept)")

# Step 4: Compute -log10(p-value) for color encoding
probit_df$logP <- -log10(probit_df$Pvalue)
probit_df$logP[is.infinite(probit_df$logP)] <- 16  # Cap for display

# Step 5: Create horizontal bar chart
ggplot(probit_df, aes(x = reorder(Feature, abs(Estimate)), y = Estimate, fill = logP)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(
    low = "lightgray",
    high = "darkred",
    name = "-log10(p-value)",
    limits = c(0, 16)
  ) +
  labs(
    title = "Probit Regression Coefficients with Significance",
    x = "Feature",
    y = "Coefficient (Probit Slope)"
  ) +
  theme_minimal(base_size = 14)