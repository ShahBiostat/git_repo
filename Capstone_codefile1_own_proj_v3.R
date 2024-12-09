rm(list=ls(all.names=TRUE))

library(tidyverse)
library(haven)
library(dplyr)
library(tidyr)
library(ggplot2)
library(psych)
library(rlang)
library(gtsummary)
library(glmulti)
library(caret)
library(purrr)
library(glmnet)
library(epiDisplay)
library(pROC)
library(ResourceSelection)
library(survey)
library(MASS)
library(boot)
library(randomForest)
library(e1071) #SVM
library(xgboost)


# STEP 1. Import data ------------------------------------------------
setwd("C:/Users/ShahS/OneDrive/EDX.ORG/DataScienceCertificate/CAPSTONE/Choose_own_proj")
nha<-read_sas("nha18_20_v3.sas7bdat")
names(nha)<-tolower(names(nha))

#inspecting data elements, sample data
head(nha)
names(nha)
str(nha)

#recoding missing values to NA
nha1<-nha%>%
  dplyr:: mutate(
    racenew_r1=ifelse(racenew_r=='', NA, racenew_r),
    health_r1=ifelse(health_r=='', NA, health_r),
    bmicat=ifelse(bmicat=='', NA, bmicat),
    poverty_r=ifelse(poverty_r=='', NA, poverty_r)
  )

#selecting appropriate variables for the summary table
nha2<- nha1%>%dplyr::select(food_insecure, food_insecure_r, year, age, bmicat, female, racenew_r1, white, black, asian, other_race,
                            hispanic, foodstamp, food_security_score, private_ins, state_ins, chi_ins, medicaid,
                            medicare, other_gov_ins, no_ins, insurance, poverty_r, health_r1, colon_scr,  cervical_scr, prostate_scr)


#Summary of demographics and clinical characteristics by group
nha2%>%
  tbl_summary(by = food_insecure_r, missing = "no") %>%
  add_p(pvalue_fun = ~style_pvalue(.x, digits = 3)) %>%
  add_overall() %>%
  add_n() %>%
  bold_labels()


##select variables for model building for colon cancer screening
nha3<- nha1%>%dplyr::select(food_insecure_r, year, age, bmicat, female, racenew_r1, insurance, poverty_r, 
                            colon_scr)

#make sure variables are recognized as factors
nha3$food_insecure_r <- factor(nha3$food_insecure_r)
nha3$racenew_r1 <- factor(nha3$racenew_r1)
nha3$female <- factor(nha3$female)
nha3$insurance <- factor(nha3$insurance)
nha3$bmicat <- factor(nha3$bmicat)
nha3$poverty_r <- factor(nha3$poverty_r)

#change reference level
nha3$food_insecure_r<-relevel(nha3$food_insecure_r, ref="1")
nha3$racenew_r1<-relevel(nha3$racenew_r1, ref="White")
nha3$female<-relevel(nha3$female, ref="0")
nha3$insurance<-relevel(nha3$insurance, ref="1")
nha3$bmicat<-relevel(nha3$bmicat, ref="2")
nha3$poverty_r<-relevel(nha3$poverty_r, ref="<1")


# Remove rows with any missing values across all columns, n=48676
nha3_colon_c <- nha3[complete.cases(nha3), ]


###Model1. Logistic Regression#######################################

model_colon=glm(colon_scr~as.factor(food_insecure_r)+age+as.factor(bmicat)+as.factor(female)+as.factor(racenew_r1)+
                  as.factor(poverty_r)+as.factor(insurance), family=binomial(link=logit),  data = nha3_colon_c)


summary(model_colon)
# Predict probabilities for each observation
predicted_probs <- predict(model_colon, nha3_colon_c, type = "response")
# Calculate AUC--model discrimination
roc_obj_logit<-roc(nha3_colon_c$colon_scr, predicted_probs)
auc_logistic <-as.numeric(roc(nha3_colon_c$colon_scr, predicted_probs)$auc)
auc_logistic

plot(roc_obj_logit, main = "ROC Curve for Logistic Regression Model")


#goodness of fit test for logistic regression
complete_cases <- complete.cases(nha3_colon_c$colon_scr, predicted_probs)
observed <- nha3_colon_c$colon_scr[complete_cases]
predicted_probs <- predicted_probs[complete_cases]
hoslem.test(observed, predicted_probs, g=10)

#Confusion Matrix and Accuracy:
predicted_probs1 <- predict(model_colon, nha3_colon_c, type = "response")
predicted <- ifelse(predicted_probs1 > 0.5, 1, 0)
conf_matrix <- table(predicted, nha3_colon_c$colon_scr)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
accuracy
sensitivity
specificity

###cross validation using caret package for logistic regression model

# Convert the outcome of interest "colon_scr" var to a factor with valid levels
#for cross validation using caret package
nha3_colon_c$colon_scr_r <- factor(
  nha3_colon_c$colon_scr, 
  levels = c(0, 1),          # Ensure correct levels
  labels = c("No", "Yes")    # Use valid string labels
)


# Define the cross-validation setup--we will use this setup for all subsequent models for cross-validation
train_control <- trainControl(
  method = "cv", 
  number = 10, 
  summaryFunction = twoClassSummary, 
  classProbs = TRUE  # Enables probability predictions
)

######cross validation for Logistic Regression model####################
cv_model <- train(
  colon_scr_r ~ as.factor(food_insecure_r) + age + as.factor(bmicat) + 
    as.factor(female) + as.factor(racenew_r1) + 
    as.factor(poverty_r) + as.factor(insurance),
  data = nha3_colon_c, 
  method = "glm", 
  family = "binomial",  # Specify logistic regression
  trControl = train_control, 
  metric = "ROC"  # Use ROC as the evaluation metric
)

# Print the model summary
print(cv_model)
summary(cv_model)

extract_auc_logistic=as.data.frame(cv_model$resample)
auc_logistic_xvalidated<-as.numeric(mean(extract_auc_logistic$ROC))

#saving AUC estimates in a dataframe
auc_table <- as.data.frame(tibble(Model ="Logistic Regression model", 
                                  AUC_single_model=auc_logistic, Auc_xvalidated = auc_logistic_xvalidated))
auc_table %>% knitr::kable()
auc_table


###2. Random Forest Model############################################################
nha3_colon_c$colon_scr <- as.factor(nha3_colon_c$colon_scr)

# Fit a random forest model
model_rf <- randomForest(colon_scr_r ~ food_insecure_r + age +bmicat +
              female + racenew_r1 +poverty_r + insurance, data = nha3_colon_c, 
              ntree = 100, importance = TRUE)


# View model summary
print(model_rf)
# Feature importance
importance(model_rf)

# Predict probabilities
pred_probs <- predict(model_rf, nha3_colon_c, type = "prob")[, 2]  # Probability of class 1
# Compute AUC
roc_obj_rf <- roc(nha3_colon_c$colon_scr, pred_probs)
auc_rf <- auc(roc_obj)
print(auc_rf)
plot(roc_obj_rf, main = "ROC Curve for Random Forest Model")

######cross validation for random forest model####################

# Train the random forest model using caret
cv_model_rf <- train(
  colon_scr_r ~ food_insecure_r + age + bmicat +
    female + racenew_r1 +
    poverty_r + insurance, 
  data = nha3_colon_c, 
  method = "rf",            # Specify Random Forest as the model
  trControl = train_control,  # Use cross-validation setup that was previously defined (see above)
  metric = "ROC",           # Optimize based on ROC AUC
  ntree = 100,              # Number of trees 
  importance = TRUE         # Track variable importance
)

# Print the model summary and plot
print(cv_model_rf)
summary(cv_model_rf)
plot(cv_model_rf, main = "Cross-Validated ROC vs. # of predictors")

extract_auc_rf=as.data.frame(cv_model_rf$resample)
auc_rf_xvalidated<-mean(extract_auc_rf$ROC)

auc_rf <- as.numeric(auc_rf)
auc_rf_xvalidated <- as.numeric(auc_rf_xvalidated)

#adding AUC to the dataframe
auc_table <- auc_table %>% add_row(Model="Random Forest Model", AUC_single_model=auc_rf, 
                                   Auc_xvalidated = auc_rf_xvalidated)
auc_table

###3. Support Vector Machine (SVM) Model########################################

model_svm <- svm(colon_scr ~ food_insecure_r + age +bmicat +
                   female + racenew_r1 +
                   poverty_r + insurance, 
                 data = nha3_colon_c, kernel = "radial", probability = TRUE)

summary(model_svm)

pred <- predict(model_svm, nha3_colon_c, probability = TRUE)
probabilities <- attr(pred, "probabilities")[, 2]  # Probabilities for the positive class
roc_obj_svm <- roc(nha3_colon_c$colon_scr, probabilities)
auc_value_svm <- as.numeric(auc(roc_obj_svm))
print(auc_value_svm)

plot(roc_obj_svm, main = "ROC Curve for SVM Model", col = "blue", lwd = 2)

######cross validation for support vector machine (SVM) model####################

# Train the SVM model using caret
cv_model_svm <- train(
  colon_scr_r ~ food_insecure_r + age + bmicat +
    female + racenew_r1 + 
    poverty_r + insurance, 
  data = nha3_colon_c, 
  method = "svmRadial",      # Specify the radial kernel SVM
  trControl = train_control, # Use cross-validation setup
  metric = "ROC",            # Optimize based on ROC AUC
  probability = TRUE         # Enable probability predictions (needed for AUC)
)

# Print the model summary
print(cv_model_svm)


extract_auc_svm=as.data.frame(cv_model_svm$resample)

auc_svm_xvalidated<-mean(as.numeric(extract_auc_svm$ROC))

#adding AUC to the dataframe
auc_table <- auc_table %>% add_row(Model="Support Vector Machine model", AUC_single_model=auc_value_svm, 
                                   Auc_xvalidated = auc_svm_xvalidated)
auc_table



###4. Gradient Boosting (e.g., XGBoost)e########################################

# Prepare data for XGBoost

nha3_colon_c$colon_scr <- as.numeric(as.character(nha3_colon_c$colon_scr))

X <- model.matrix(colon_scr ~ food_insecure_r + age +bmicat +
                    female + racenew_r1 +
                    poverty_r + insurance, 
                  data = nha3_colon_c)[,-1]
y <- nha3_colon_c$colon_scr

# Train-test split
set.seed(1157)
train_idx <- sample(1:nrow(nha3_colon_c), 0.8 * nrow(nha3_colon_c))

dtrain <- xgb.DMatrix(data = X[train_idx,], label = y[train_idx])
dtest <- xgb.DMatrix(data = X[-train_idx,], label = y[-train_idx])

# dtrain_labels <- as.numeric(as.factor(your_labels)) - 1
# dtrain <- xgb.DMatrix(data = your_features, label = dtrain_labels)


# Train XGBoost
model_xgb <- xgboost(data = dtrain, objective = "binary:logistic", nrounds = 100)

# Predict on test data
preds <- predict(model_xgb, dtest)
# Extract true labels from the DMatrix
true_labels <- getinfo(dtest, "label")

# Compute the ROC curve
roc_obj_xgb <- roc(true_labels, preds)

# Compute the AUC
auc_value_xgb <- as.numeric(auc(roc_obj_xgb))
print(auc_value_xgb)

# Plot the ROC curve
plot(auc_value_xgb, main = "ROC Curve for XGBoost Model", col = "blue", lwd = 2)

######cross validation for XGBoost model####################

# Train the XGBoost model using caret
cv_model_xgb <- train(
  colon_scr_r ~ food_insecure_r + age + bmicat +
    female + racenew_r1 + 
    poverty_r + insurance, 
  data = nha3_colon_c, 
  method = "xgbTree",       # Specify XGBoost as the method
  trControl = train_control, # Use cross-validation setup
  metric = "ROC",           # Optimize based on ROC AUC
  tuneLength = 3            # Test 3 random parameter combinations (optional)
)

# Print the model summary
print(cv_model_xgb)

plot(cv_model_xgb, main = "XGBoost ROC vs. # of predictors")

extract_auc_xgb=as.data.frame(cv_model_xgb$resample)

auc_xgb_xvalidated<-mean(as.numeric(extract_auc_xgb$ROC))

#adding AUC metric to the dataframe
auc_table <- auc_table %>% add_row(Model="XGBoost model", AUC_single_model=auc_value_xgb, 
                                   Auc_xvalidated = auc_xgb_xvalidated)
auc_table





