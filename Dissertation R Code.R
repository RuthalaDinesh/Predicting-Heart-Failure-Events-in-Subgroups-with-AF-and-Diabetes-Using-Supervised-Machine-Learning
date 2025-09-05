#### Load Required Libraries ####
library(tidyverse)
library(caret)
library(xgboost)
library(e1071)
library(pROC)
library(PRROC)
library(corrplot)
library(knitr)
library(kableExtra)
library(dplyr)
library(forcats)
library(tibble)
library(kernlab)
library(ggplot2)

df <- read.csv("data_aurum.csv")
df
head(df)

df <- df %>% 
  select(X_case, depression, copd, mi, valve, af, htn, vascular, gender, age_index, diabetes) %>% 
  view()

view(df)
df

df$X_case <- as.factor(df$X_case)
class(df$X_case)

df$gender <- as.factor(df$gender)
class(df$gender)

df <- mutate(df, X_case = recode(X_case,
                                 control = "0",
                                 case  = "1")) %>% 
  mutate(gender = recode(gender,
                         M = "0",
                         F = "1"))
view(df)

view(df)
df

sum(is.na(df))

table(df$X_case)

# AF Subgroup
df_af <- df %>% filter(af == 1)
df_non_af <- df %>% filter(af == 0)

view(df_af)

# Diabetes Subgroup
df_diabetes <- df %>% filter(diabetes == 1)
df_non_diabetes <- df %>% filter(diabetes == 0)

# Gender Subgroup
df_female <- df %>% filter(gender == 1)
df_male <- df %>% filter(gender == 0)


#### Ensure outcome as factor and Use the same labels ####
df$X_case  <- factor(df$X_case,  levels = c(0, 1), labels = c("NonHF", "HF"))

df_af$X_case  <- factor(df_af$X_case,  levels = c(0, 1), labels = c("NonHF", "HF"))
df_non_af$X_case  <- factor(df_non_af$X_case,  levels = c(0, 1), labels = c("NonHF", "HF"))

df_diabetes$X_case  <- factor(df_diabetes$X_case,  levels = c(0, 1), labels = c("NonHF", "HF"))
df_non_diabetes$X_case  <- factor(df_non_diabetes$X_case,  levels = c(0, 1), labels = c("NonHF", "HF"))

df_female$X_case  <- factor(df_female$X_case,  levels = c(0, 1), labels = c("NonHF", "HF"))
df_male$X_case  <- factor(df_male$X_case,  levels = c(0, 1), labels = c("NonHF", "HF"))


#### Train-Test Split ####
#### Load Required Libraries ####
library(dplyr)
library(tidyr)
library(caret)
library(ROCR)
library(pROC)
library(PRROC)
library(e1071)
library(xgboost)
library(Matrix)
library(tibble)
library(ggplot2)
library(parallel)
library(doParallel)


##### Split the dataset into Training and Testing by 50:50 #####
set.seed(235)
train_index <- createDataPartition(df$X_case, p = 0.5, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]



##### Train Logistic Regression Model #####
cv_control_logit <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

logit_model <- train(
  X_case ~ ., 
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = cv_control_logit
)

##### Parallel Setup for XGBoost and SVM #####
cluster <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cluster)

cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE
)

##### Train XGBoost Model #####
# Convert factors to numeric matrix for xgboost
xgb_train <- train_data
xgb_train$X_case <- ifelse(xgb_train$X_case == "HF", 1, 0)
xgb_matrix <- xgb.DMatrix(data = model.matrix(~ . -1, data = xgb_train %>% select(-X_case)), 
                          label = xgb_train$X_case)

xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = 3,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)


xgb_model <- train(
  X_case ~ ., 
  data = train_data,
  method = "xgbTree",
  trControl = cv_control,
  tuneGrid = xgb_grid,
  metric = "ROC"
)

stopCluster(cluster)

##### Find Optimal Thresholds using Youden Index #####
roc_logit <- roc(test_data$X_case, predict(logit_model, newdata = test_data, type = "prob")[, "HF"])
thresh_logit <- coords(roc_logit, "best", ret = "threshold")

roc_xgb <- roc(test_data$X_case, predict(xgb_model, newdata = test_data, type = "prob")[, "HF"])
thresh_xgb <- coords(roc_xgb, "best", ret = "threshold")

##### Evaluation Function #####
evaluate_on_test <- function(model, test_data, threshold = 0.5) {
  probs <- predict(model, newdata = test_data, type = "prob")[, "HF"]
  preds <- ifelse(probs > threshold, "HF", "NonHF")
  
  cm <- confusionMatrix(factor(preds, levels = c("NonHF", "HF")), test_data$X_case, positive = "HF")
  
  roc_obj <- roc(test_data$X_case, probs)
  pr_obj <- pr.curve(scores.class0 = probs[test_data$X_case == "HF"],
                     scores.class1 = probs[test_data$X_case == "NonHF"],
                     curve = FALSE)
  
  target_numeric <- as.numeric(test_data$X_case) - 1
  brier_score <- mean((probs - target_numeric)^2)
  
  eps <- 1e-15
  probs <- pmin(pmax(probs, eps), 1 - eps)
  logit_probs <- log(probs / (1 - probs))
  calib_model <- glm(target_numeric ~ logit_probs, family = "binomial")
  calib_slope <- coef(calib_model)["logit_probs"]
  
  list(
    ConfusionMatrix = cm,
    AUROC = auc(roc_obj),
    PRAUC = pr_obj$auc.integral,
    BrierScore = brier_score,
    CalibrationSlope = calib_slope
  )
}

##### Evaluate Models on Test Set #####
logit_eval <- evaluate_on_test(logit_model, test_data, threshold = 0.00962446)
xgb_eval <- evaluate_on_test(xgb_model, test_data, threshold = 0.01069671)

##### Create Comparison Table #####
model_comparison <- bind_rows(
  tibble(
    Model = "Logistic Regression",
    AUROC = as.numeric(logit_eval$AUROC),
    PRAUC = logit_eval$PRAUC,
    Sensitivity = logit_eval$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval$BrierScore,
    CalibrationSlope = logit_eval$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost",
    AUROC = as.numeric(xgb_eval$AUROC),
    PRAUC = xgb_eval$PRAUC,
    Sensitivity = xgb_eval$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval$BrierScore,
    CalibrationSlope = xgb_eval$CalibrationSlope
  )
)

print(model_comparison)


##### Plot ROC Curve #####
png("AUC-ROC-Comparison.png", width = 1200, height = 800)

roc_logit <- roc(test_data$X_case, predict(logit_model, newdata = test_data, type = "prob")[, "HF"])
plot(roc_logit, col = "blue", lwd = 2, main = "ROC Curve Comparison", legacy.axes = TRUE)

roc_xgb <- roc(test_data$X_case, predict(xgb_model, newdata = test_data, type = "prob")[, "HF"])
lines(roc_xgb, col = "green", lwd = 2)

legend("bottomright",
       legend = c(
         paste0("Logistic Regression (AUC = ", round(logit_eval$AUROC, 3), ")"),
         paste0("XGBoost (AUC = ", round(xgb_eval$AUROC, 3), ")")
       ),
       col = c("blue", "green"), lwd = 2)

dev.off()

##### Plot Precision-Recall Curve #####
png("PR-AUC-Comparison.png", width = 1200, height = 800)

# Logistic Regression
probs_logit <- predict(logit_model, newdata = test_data, type = "prob")[, "HF"]
pr_logit <- pr.curve(scores.class0 = probs_logit[test_data$X_case == "HF"],
                     scores.class1 = probs_logit[test_data$X_case == "NonHF"],
                     curve = TRUE)
plot(pr_logit, col = "blue", lwd = 2, main = "Precision-Recall Curve Comparison")

# XGBoost
probs_xgb <- predict(xgb_model, newdata = test_data, type = "prob")[, "HF"]
pr_xgb <- pr.curve(scores.class0 = probs_xgb[test_data$X_case == "HF"],
                   scores.class1 = probs_xgb[test_data$X_case == "NonHF"],
                   curve = TRUE)
lines(pr_xgb$curve[,1], pr_xgb$curve[,2], col = "green", lwd = 2)

legend("topright",
       legend = c(
         paste0("Logistic Regression (AUC = ", round(logit_eval$PRAUC, 3), ")"),
         paste0("XGBoost (AUC = ", round(xgb_eval$PRAUC, 3), ")")
       ),
       col = c("blue", "green"), lwd = 2)

dev.off()



#### Refit the Best Model using full data set ####
# Fit the XGBoost model using xgboost() function

# Prepare data for XGBoost training
data3_xgb <- df
# Convert outcome to numeric 0/1 for XGBoost
data3_xgb$X_case <- ifelse(data3_xgb$X_case == "HF", 1, 0)

# Create xgboost matrix
xgb_matrix_full <- xgb.DMatrix(
  data = model.matrix(~ . -1, data = data3_xgb %>% select(-X_case)),
  label = data3_xgb$X_case
)

# Set parameters (adjust if you tuned differently)
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 3,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

# Train final XGBoost model on full data
xgb_fitted <- xgboost(
  params = xgb_params,
  data = xgb_matrix_full,
  nrounds = 100,
  verbose = 0
)

#### Evaluate Fitted model's Performance ####

evaluate_xgb_fitted <- function(model, test_data, threshold = 0.5) {
  # test_data must have factor X_case with levels c("NonHF", "HF")
  
  # Convert outcome to numeric for Brier score and calibration slope
  target_numeric <- ifelse(test_data$X_case == "HF", 1, 0)
  
  # Prepare test matrix for prediction
  xgb_matrix_test <- xgb.DMatrix(data = model.matrix(~ . -1, data = test_data %>% select(-X_case)))
  
  # Predict probabilities
  probs <- predict(model, newdata = xgb_matrix_test)
  
  # Predict classes based on threshold
  preds <- ifelse(probs > threshold, "HF", "NonHF")
  
  # Confusion matrix
  cm <- confusionMatrix(
    factor(preds, levels = c("NonHF", "HF")),
    test_data$X_case,
    positive = "HF"
  )
  
  # ROC and PR curves
  roc_obj <- roc(test_data$X_case, probs)
  pr_obj <- pr.curve(
    scores.class0 = probs[test_data$X_case == "HF"],
    scores.class1 = probs[test_data$X_case == "NonHF"],
    curve = FALSE
  )
  
  # Brier score
  brier_score <- mean((probs - target_numeric)^2)
  
  # Calibration slope
  eps <- 1e-15
  probs <- pmin(pmax(probs, eps), 1 - eps)
  logit_probs <- log(probs / (1 - probs))
  calib_model <- glm(target_numeric ~ logit_probs, family = "binomial")
  calib_slope <- coef(calib_model)["logit_probs"]
  
  list(
    ConfusionMatrix = cm,
    AUROC = auc(roc_obj),
    PRAUC = pr_obj$auc.integral,
    BrierScore = brier_score,
    CalibrationSlope = calib_slope
  )
}

##### Evaluate Fitted Model on Subgroup Dataset #####

# Make sure all datasets have factor X_case with correct levels
df$X_case <- factor(df$X_case, levels = c("NonHF", "HF"))

df_af$X_case <- factor(df_af$X_case, levels = c("NonHF", "HF"))
df_non_af$X_case <- factor(df_non_af$X_case, levels = c("NonHF", "HF"))

df_diabetes$X_case <- factor(df_diabetes$X_case, levels = c("NonHF", "HF"))
df_non_diabetes$X_case <- factor(df_non_diabetes$X_case, levels = c("NonHF", "HF"))

df_female$X_case <- factor(df_female$X_case, levels = c("NonHF", "HF"))
df_male$X_case <- factor(df_male$X_case, levels = c("NonHF", "HF"))

# Evaluate on entire dataset
xgb_eval_fitted <- evaluate_xgb_fitted(xgb_fitted, df, threshold = 0.01069671)
xgb_eval_fitted

# Evaluate on AF subgroup
xgb_eval_af <- evaluate_xgb_fitted(xgb_fitted, df_af, threshold = 0.01069671)
xgb_eval_non_af <- evaluate_xgb_fitted(xgb_fitted, df_non_af, threshold = 0.01069671)

# Evaluate on Diabetes subgroup
xgb_eval_diabetes <- evaluate_xgb_fitted(xgb_fitted, df_diabetes, threshold = 0.01069671)
xgb_eval_non_diabetes <- evaluate_xgb_fitted(xgb_fitted, df_non_diabetes, threshold = 0.01069671)

# Evaluate on Gender subgroup
xgb_eval_female <- evaluate_xgb_fitted(xgb_fitted, df_female, threshold = 0.01069671)
xgb_eval_male <- evaluate_xgb_fitted(xgb_fitted, df_male, threshold = 0.01069671)

##### Create comparison tibble #####
model_comparison_fitted_xgb <- bind_rows(
  tibble(
    Model = "XGBoost",
    AUROC = as.numeric(xgb_eval_fitted$AUROC),
    PRAUC = xgb_eval_fitted$PRAUC,
    Sensitivity = xgb_eval_fitted$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_fitted$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_fitted$BrierScore,
    CalibrationSlope = xgb_eval_fitted$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (AF)",
    AUROC = as.numeric(xgb_eval_af$AUROC),
    PRAUC = xgb_eval_af$PRAUC,
    Sensitivity = xgb_eval_af$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_af$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_af$BrierScore,
    CalibrationSlope = xgb_eval_af$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Non AF)",
    AUROC = as.numeric(xgb_eval_non_af$AUROC),
    PRAUC = xgb_eval_non_af$PRAUC,
    Sensitivity = xgb_eval_non_af$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_non_af$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_non_af$BrierScore,
    CalibrationSlope = xgb_eval_non_af$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Diabetes)",
    AUROC = as.numeric(xgb_eval_diabetes$AUROC),
    PRAUC = xgb_eval_diabetes$PRAUC,
    Sensitivity = xgb_eval_diabetes$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_diabetes$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_diabetes$BrierScore,
    CalibrationSlope = xgb_eval_diabetes$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Non Diabetes)",
    AUROC = as.numeric(xgb_eval_non_diabetes$AUROC),
    PRAUC = xgb_eval_non_diabetes$PRAUC,
    Sensitivity = xgb_eval_non_diabetes$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_non_diabetes$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_non_diabetes$BrierScore,
    CalibrationSlope = xgb_eval_non_diabetes$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Female)",
    AUROC = as.numeric(xgb_eval_female$AUROC),
    PRAUC = xgb_eval_female$PRAUC,
    Sensitivity = xgb_eval_female$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_female$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_female$BrierScore,
    CalibrationSlope = xgb_eval_female$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Male)",
    AUROC = as.numeric(xgb_eval_male$AUROC),
    PRAUC = xgb_eval_male$PRAUC,
    Sensitivity = xgb_eval_male$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_male$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_male$BrierScore,
    CalibrationSlope = xgb_eval_male$CalibrationSlope
  )
)

print(model_comparison_fitted_xgb)
view(print(model_comparison_fitted_xgb))



# ROC and PR curves for subgroup datasets (Global XGBoost on subgroups)


library(pROC)
library(PRROC)
library(ggplot2)

### Helper function for ROC + PR plotting
plot_subgroup_curves_global <- function(model, data, group_name) {
  # Convert predictors into numeric matrix
  X <- model.matrix(~ . -1, data = data %>% select(-X_case))
  y <- as.numeric(as.factor(data$X_case)) - 1   # force 0/1
  
  # DMatrix for XGBoost
  dmatrix <- xgb.DMatrix(data = X, label = y)
  
  # Predictions
  preds <- predict(model, dmatrix)
  
  # ROC
  roc_obj <- roc(y, preds)
  auc_val <- auc(roc_obj)
  roc_plot <- ggroc(roc_obj, color = "darkblue", size = 1.2) +
    ggtitle(paste("ROC Curve -", group_name, "\nAUROC =", round(auc_val, 3))) +
    theme_minimal()
  
  # PR with safeguard
  if (length(unique(y)) < 2) {
    pr_plot <- ggplot() +
      ggtitle(paste("PR Curve -", group_name, "\n(Not enough positives/negatives)")) +
      theme_void()
  } else {
    pr_obj <- pr.curve(
      scores.class0 = preds[y == 1],
      scores.class1 = preds[y == 0],
      curve = TRUE
    )
    pr_data <- data.frame(Recall = pr_obj$curve[,1],
                          Precision = pr_obj$curve[,2])
    
    pr_plot <- ggplot(pr_data, aes(x = Recall, y = Precision)) +
      geom_line(color = "darkred", size = 1.2) +
      ggtitle(paste("PR Curve -", group_name, "\nAUPRC =", round(pr_obj$auc.integral, 3))) +
      theme_minimal()
  }
  
  return(list(roc_plot = roc_plot, pr_plot = pr_plot))
}



### Run for each subgroup (global model applied)
plots_af <- plot_subgroup_curves_global(xgb_fitted, df_af, "AF")
plots_non_af <- plot_subgroup_curves_global(xgb_fitted, df_non_af, "Non-AF")

plots_diabetes <- plot_subgroup_curves_global(xgb_fitted, df_diabetes, "Diabetes")
plots_non_diabetes <- plot_subgroup_curves_global(xgb_fitted, df_non_diabetes, "Non-Diabetes")

plots_female <- plot_subgroup_curves_global(xgb_fitted, df_female, "Female")
plots_male <- plot_subgroup_curves_global(xgb_fitted, df_male, "Male")

### Example: display AF subgroup ROC + PR
print(plots_af$roc_plot)
print(plots_af$pr_plot)

### You can print the rest as needed:
print(plots_non_af$roc_plot); print(plots_non_af$pr_plot)
print(plots_diabetes$roc_plot); print(plots_diabetes$pr_plot)
print(plots_non_diabetes$roc_plot); print(plots_non_diabetes$pr_plot)
print(plots_female$roc_plot); print(plots_female$pr_plot)
print(plots_male$roc_plot); print(plots_male$pr_plot)






#### Refit the Best Model using Subgroup data set ####

# Prepare subgroup datasets by removing subgroup indicator variable (e.g., 'af')
data_af <- df_af %>% select(-af)
data_non_af <- df_non_af %>% select(-af)

# Convert outcome to numeric 0/1 for XGBoost
data_af_xgb <- data_af
data_af_xgb$X_case <- ifelse(data_af_xgb$X_case == "HF", 1, 0)

data_non_af_xgb <- data_non_af
data_non_af_xgb$X_case <- ifelse(data_non_af_xgb$X_case == "HF", 1, 0)

# Create DMatrix for XGBoost
xgb_matrix_af <- xgb.DMatrix(
  data = model.matrix(~ . -1, data = data_af_xgb %>% select(-X_case)),
  label = data_af_xgb$X_case
)

xgb_matrix_non_af <- xgb.DMatrix(
  data = model.matrix(~ . -1, data = data_non_af_xgb %>% select(-X_case)),
  label = data_non_af_xgb$X_case
)

# Define parameters (can keep same as before)
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 3,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

# Fit XGBoost model on AF subgroup
xgb_fitted_af <- xgboost(
  params = xgb_params,
  data = xgb_matrix_af,
  nrounds = 100,
  verbose = 0
)

# Fit XGBoost model on Non-AF subgroup
xgb_fitted_non_af <- xgboost(
  params = xgb_params,
  data = xgb_matrix_non_af,
  nrounds = 100,
  verbose = 0
)


#### Refit the Best Model using Diabetes Subgroup data set ####

# Remove the 'diabetes' column from subgroup datasets
data_diabetes <- df_diabetes %>% select(-diabetes)
data_non_diabetes <- df_non_diabetes %>% select(-diabetes)

# Convert outcome to numeric 0/1 for XGBoost
data_diabetes_xgb <- data_diabetes
data_diabetes_xgb$X_case <- ifelse(data_diabetes_xgb$X_case == "HF", 1, 0)

data_non_diabetes_xgb <- data_non_diabetes
data_non_diabetes_xgb$X_case <- ifelse(data_non_diabetes_xgb$X_case == "HF", 1, 0)

# Create DMatrix for XGBoost
xgb_matrix_diabetes <- xgb.DMatrix(
  data = model.matrix(~ . -1, data = data_diabetes_xgb %>% select(-X_case)),
  label = data_diabetes_xgb$X_case
)

xgb_matrix_non_diabetes <- xgb.DMatrix(
  data = model.matrix(~ . -1, data = data_non_diabetes_xgb %>% select(-X_case)),
  label = data_non_diabetes_xgb$X_case
)

# Fit XGBoost model on Diabetes subgroup
xgb_fitted_diabetes <- xgboost(
  params = xgb_params,
  data = xgb_matrix_diabetes,
  nrounds = 100,
  verbose = 0
)

# Fit XGBoost model on Non-Diabetes subgroup
xgb_fitted_non_diabetes <- xgboost(
  params = xgb_params,
  data = xgb_matrix_non_diabetes,
  nrounds = 100,
  verbose = 0
)


#### Refit the Best Model using Gender Subgroup data set ####

# Remove the 'gender' column from subgroup datasets
data_female <- df_female %>% select(-gender)
data_male <- df_male %>% select(-gender)

# Convert outcome to numeric 0/1 for XGBoost
data_female_xgb <- data_female
data_female_xgb$X_case <- ifelse(data_female_xgb$X_case == "HF", 1, 0)

data_male_xgb <- data_male
data_male_xgb$X_case <- ifelse(data_male_xgb$X_case == "HF", 1, 0)

# Create DMatrix for XGBoost
xgb_matrix_female <- xgb.DMatrix(
  data = model.matrix(~ . -1, data = data_female_xgb %>% select(-X_case)),
  label = data_female_xgb$X_case
)

xgb_matrix_male <- xgb.DMatrix(
  data = model.matrix(~ . -1, data = data_male_xgb %>% select(-X_case)),
  label = data_male_xgb$X_case
)

# Fit XGBoost model on Female subgroup
xgb_fitted_female <- xgboost(
  params = xgb_params,
  data = xgb_matrix_female,
  nrounds = 100,
  verbose = 0
)

# Fit XGBoost model on Male subgroup
xgb_fitted_male <- xgboost(
  params = xgb_params,
  data = xgb_matrix_male,
  nrounds = 100,
  verbose = 0
)

#### Evaluate Second Fitted Model on Subgroup Dataset ####
# Ensure factor levels are correct
df_af$X_case <- factor(df_af$X_case, levels = c("NonHF", "HF"))
df_non_af$X_case <- factor(df_non_af$X_case, levels = c("NonHF", "HF"))

# Drop 'af' column to match training data structure
df_af_xgb <- df_af %>% select(-af)
df_non_af_xgb <- df_non_af %>% select(-af)

# Reorder columns to match the training dataset used for model fitting
df_af_xgb <- df_af_xgb %>% select(all_of(colnames(data_af_xgb)))
df_non_af_xgb <- df_non_af_xgb %>% select(all_of(colnames(data_non_af_xgb)))

# Evaluate using the fitted XGBoost models and threshold
xgb_eval_af2 <- evaluate_xgb_fitted(xgb_fitted_af, df_af_xgb, threshold = 0.01069671)
xgb_eval_non_af2 <- evaluate_xgb_fitted(xgb_fitted_non_af, df_non_af_xgb, threshold = 0.01069671)


#### Evaluate Second Fitted Model on Diabetes Subgroup ####
# Ensure factor levels are correct
df_diabetes$X_case <- factor(df_diabetes$X_case, levels = c("NonHF", "HF"))
df_non_diabetes$X_case <- factor(df_non_diabetes$X_case, levels = c("NonHF", "HF"))

# Drop 'diabetes' column to match model input structure
df_diabetes_xgb <- df_diabetes %>% select(-diabetes)
df_non_diabetes_xgb <- df_non_diabetes %>% select(-diabetes)

# Reorder columns to match model training structure
df_diabetes_xgb <- df_diabetes_xgb %>% select(all_of(colnames(data_diabetes_xgb)))
df_non_diabetes_xgb <- df_non_diabetes_xgb %>% select(all_of(colnames(data_non_diabetes_xgb)))

# Evaluate XGBoost fitted models
xgb_eval_diabetes2 <- evaluate_xgb_fitted(xgb_fitted_diabetes, df_diabetes_xgb, threshold = 0.01069671)
xgb_eval_non_diabetes2 <- evaluate_xgb_fitted(xgb_fitted_non_diabetes, df_non_diabetes_xgb, threshold = 0.01069671)


#### Evaluate Second Fitted Model on Gender Subgroup ####
# Ensure factor levels are correct
df_female$X_case <- factor(df_female$X_case, levels = c("NonHF", "HF"))
df_male$X_case <- factor(df_male$X_case, levels = c("NonHF", "HF"))

# Drop 'gender' column to match model input structure
df_female_xgb <- df_female %>% select(-gender)
df_male_xgb <- df_male %>% select(-gender)

# Reorder columns to match model training structure
df_female_xgb <- df_female_xgb %>% select(all_of(colnames(data_female_xgb)))
df_male_xgb <- df_male_xgb %>% select(all_of(colnames(data_male_xgb)))

# Evaluate XGBoost fitted models
xgb_eval_female2 <- evaluate_xgb_fitted(xgb_fitted_female, df_female_xgb, threshold = 0.01069671)
xgb_eval_male2 <- evaluate_xgb_fitted(xgb_fitted_male, df_male_xgb, threshold = 0.01069671)


# Create comparison tibble for XGBoost model
model_comparison_fitted_xgb2 <- bind_rows(
  tibble(
    Model = "XGBoost",
    AUROC = as.numeric(xgb_eval_fitted$AUROC),
    PRAUC = xgb_eval_fitted$PRAUC,
    Sensitivity = xgb_eval_fitted$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_fitted$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_fitted$BrierScore,
    CalibrationSlope = xgb_eval_fitted$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (AF)",
    AUROC = as.numeric(xgb_eval_af2$AUROC),
    PRAUC = xgb_eval_af2$PRAUC,
    Sensitivity = xgb_eval_af2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_af2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_af2$BrierScore,
    CalibrationSlope = xgb_eval_af2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Non AF)",
    AUROC = as.numeric(xgb_eval_non_af2$AUROC),
    PRAUC = xgb_eval_non_af2$PRAUC,
    Sensitivity = xgb_eval_non_af2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_non_af2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_non_af2$BrierScore,
    CalibrationSlope = xgb_eval_non_af2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Diabetes)",
    AUROC = as.numeric(xgb_eval_diabetes2$AUROC),
    PRAUC = xgb_eval_diabetes2$PRAUC,
    Sensitivity = xgb_eval_diabetes2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_diabetes2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_diabetes2$BrierScore,
    CalibrationSlope = xgb_eval_diabetes2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Non Diabetes)",
    AUROC = as.numeric(xgb_eval_non_diabetes2$AUROC),
    PRAUC = xgb_eval_non_diabetes2$PRAUC,
    Sensitivity = xgb_eval_non_diabetes2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_non_diabetes2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_non_diabetes2$BrierScore,
    CalibrationSlope = xgb_eval_non_diabetes2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Female)",
    AUROC = as.numeric(xgb_eval_female2$AUROC),
    PRAUC = xgb_eval_female2$PRAUC,
    Sensitivity = xgb_eval_female2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_female2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_female2$BrierScore,
    CalibrationSlope = xgb_eval_female2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Male)",
    AUROC = as.numeric(xgb_eval_male2$AUROC),
    PRAUC = xgb_eval_male2$PRAUC,
    Sensitivity = xgb_eval_male2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_male2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_male2$BrierScore,
    CalibrationSlope = xgb_eval_male2$CalibrationSlope
  )
)

# Show comparison
print(model_comparison_fitted_xgb2)
view(print(model_comparison_fitted_xgb2))



# If threshold for entire data is not used for evaluating each subgroup, then we find optimal threshold values for each subgroup and then evaluate accordingly. 

get_optimal_threshold <- function(model, data) {
  # Ensure outcome is binary numeric: HF = 1, NonHF = 0
  actual <- ifelse(data$X_case == "HF", 1, 0)
  data_matrix <- xgb.DMatrix(data = model.matrix(X_case ~ . - 1, data = data))
  
  probs <- predict(model, data_matrix)
  pred <- ROCR::prediction(probs, actual)
  perf <- ROCR::performance(pred, "sens", "spec")
  
  sens <- perf@y.values[[1]]
  spec <- perf@x.values[[1]]
  thresholds <- perf@alpha.values[[1]]
  
  youden_index <- sens + spec - 1
  best_threshold <- thresholds[which.max(youden_index)]
  
  return(best_threshold)
}


threshold_af <- get_optimal_threshold(xgb_fitted_af, df_af_xgb)
xgb_eval_af2 <- evaluate_xgb_fitted(xgb_fitted_af, df_af_xgb, threshold = threshold_af)

threshold_non_af <- get_optimal_threshold(xgb_fitted_non_af, df_non_af_xgb)
xgb_eval_non_af2 <- evaluate_xgb_fitted(xgb_fitted_non_af, df_non_af_xgb, threshold = threshold_non_af)

threshold_af
threshold_non_af


threshold_diabetes <- get_optimal_threshold(xgb_fitted_diabetes, df_diabetes_xgb)
xgb_eval_diabetes2 <- evaluate_xgb_fitted(xgb_fitted_diabetes, df_diabetes_xgb, threshold = threshold_diabetes)

threshold_non_diabetes <- get_optimal_threshold(xgb_fitted_non_diabetes, df_non_diabetes_xgb)
xgb_eval_non_diabetes2 <- evaluate_xgb_fitted(xgb_fitted_non_diabetes, df_non_diabetes_xgb, threshold = threshold_non_diabetes)

threshold_diabetes
threshold_non_diabetes


threshold_female <- get_optimal_threshold(xgb_fitted_female, df_female_xgb)
xgb_eval_female2 <- evaluate_xgb_fitted(xgb_fitted_female, df_female_xgb, threshold = threshold_female)

threshold_male <- get_optimal_threshold(xgb_fitted_male, df_male_xgb)
xgb_eval_male2 <- evaluate_xgb_fitted(xgb_fitted_male, df_male_xgb, threshold = threshold_male)

threshold_female
threshold_male


xgb_eval_af2
xgb_eval_non_af2
xgb_eval_diabetes2
xgb_eval_non_diabetes2
xgb_eval_female2
xgb_eval_male2


model_comparison_fitted_xgb2 <- bind_rows(
  tibble(
    Model = "XGBoost",
    AUROC = as.numeric(xgb_eval_fitted$AUROC),
    PRAUC = xgb_eval_fitted$PRAUC,
    Sensitivity = xgb_eval_fitted$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_fitted$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_fitted$BrierScore,
    CalibrationSlope = xgb_eval_fitted$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (AF)",
    AUROC = as.numeric(xgb_eval_af2$AUROC),
    PRAUC = xgb_eval_af2$PRAUC,
    Sensitivity = xgb_eval_af2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_af2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_af2$BrierScore,
    CalibrationSlope = xgb_eval_af2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Non AF)",
    AUROC = as.numeric(xgb_eval_non_af2$AUROC),
    PRAUC = xgb_eval_non_af2$PRAUC,
    Sensitivity = xgb_eval_non_af2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_non_af2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_non_af2$BrierScore,
    CalibrationSlope = xgb_eval_non_af2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Diabetes)",
    AUROC = as.numeric(xgb_eval_diabetes2$AUROC),
    PRAUC = xgb_eval_diabetes2$PRAUC,
    Sensitivity = xgb_eval_diabetes2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_diabetes2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_diabetes2$BrierScore,
    CalibrationSlope = xgb_eval_diabetes2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Non Diabetes)",
    AUROC = as.numeric(xgb_eval_non_diabetes2$AUROC),
    PRAUC = xgb_eval_non_diabetes2$PRAUC,
    Sensitivity = xgb_eval_non_diabetes2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_non_diabetes2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_non_diabetes2$BrierScore,
    CalibrationSlope = xgb_eval_non_diabetes2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Female)",
    AUROC = as.numeric(xgb_eval_female2$AUROC),
    PRAUC = xgb_eval_female2$PRAUC,
    Sensitivity = xgb_eval_female2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_female2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_female2$BrierScore,
    CalibrationSlope = xgb_eval_female2$CalibrationSlope
  ),
  tibble(
    Model = "XGBoost (Male)",
    AUROC = as.numeric(xgb_eval_male2$AUROC),
    PRAUC = xgb_eval_male2$PRAUC,
    Sensitivity = xgb_eval_male2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = xgb_eval_male2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = xgb_eval_male2$BrierScore,
    CalibrationSlope = xgb_eval_male2$CalibrationSlope
  )
)

# Show comparison
print(model_comparison_fitted_xgb2)
view(print(model_comparison_fitted_xgb2))




# ROC and PR curves for subgroups of data (Subgroup-specific modelling)

library(ggplot2)
library(pROC)
library(PRROC)
library(dplyr)
library(tibble)

plot_roc_pr_subgroups <- function(model, datasets, thresholds, subgroup_names) {
  
  roc_list <- list()
  pr_list <- list()
  
  for (i in seq_along(datasets)) {
    data <- datasets[[i]]
    threshold <- thresholds[i]
    name <- subgroup_names[i]
    
    # Prepare XGBoost matrix
    xgb_mat <- xgb.DMatrix(data = model.matrix(~ . -1, data = data %>% select(-X_case)))
    probs <- predict(model, newdata = xgb_mat)
    
    # ROC
    roc_obj <- roc(data$X_case, probs)
    roc_df <- tibble(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities,
      Model = name
    )
    roc_list[[i]] <- roc_df
    
    # PR curve
    pr_obj <- pr.curve(
      scores.class0 = probs[data$X_case == "HF"],
      scores.class1 = probs[data$X_case == "NonHF"],
      curve = TRUE
    )
    pr_df <- tibble(
      recall = pr_obj$curve[,1],
      precision = pr_obj$curve[,2],
      Model = name
    )
    pr_list[[i]] <- pr_df
  }
  
  # Combine all subgroups
  roc_all <- bind_rows(roc_list)
  pr_all <- bind_rows(pr_list)
  
  # ROC plot
  p_roc <- ggplot(roc_all, aes(x = fpr, y = tpr, color = Model)) +
    geom_line(size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
    labs(title = "ROC Curves - XGBoost Subgroups", x = "False Positive Rate", y = "True Positive Rate") +
    theme_minimal() +
    theme(text = element_text(size = 14))
  
  # PR plot
  p_pr <- ggplot(pr_all, aes(x = recall, y = precision, color = Model)) +
    geom_line(size = 1.2) +
    labs(title = "Precision-Recall Curves - XGBoost Subgroups", x = "Recall", y = "Precision") +
    theme_minimal() +
    theme(text = element_text(size = 14))
  
  list(ROC = p_roc, PR = p_pr)
}

# Example usage for AF subgroup
datasets <- list(df_af_xgb, df_non_af_xgb)
thresholds <- c(threshold_af, threshold_non_af)
subgroup_names <- c("AF", "Non-AF")

plots_af <- plot_roc_pr_subgroups(xgb_fitted_af, datasets, thresholds, subgroup_names)
plots_af$ROC
plots_af$PR

# Similarly for Diabetes
datasets <- list(df_diabetes_xgb, df_non_diabetes_xgb)
thresholds <- c(threshold_diabetes, threshold_non_diabetes)
subgroup_names <- c("Diabetes", "Non-Diabetes")

plots_diabetes <- plot_roc_pr_subgroups(xgb_fitted_diabetes, datasets, thresholds, subgroup_names)
plots_diabetes$ROC
plots_diabetes$PR

# Similarly for Gender
datasets <- list(df_female_xgb, df_male_xgb)
thresholds <- c(threshold_female, threshold_male)
subgroup_names <- c("Female", "Male")

plots_gender <- plot_roc_pr_subgroups(xgb_fitted_female, datasets, thresholds, subgroup_names)
plots_gender$ROC
plots_gender$PR





view(probs_logit)
preds_logit <- ifelse(probs_logit > 0.009624468, "HF", "NonHF")
preds_logit

view(probs_xgb)
preds_xgb <- ifelse(probs_xgb > 0.01069671, "HF", "NonHF")
preds_xgb




logit_eval


table(ifelse(probs_logit > 0.5, "HF", "NonHF"))
table(preds_logit)


logit_eval
xgb_eval











# ---- Predicted outcome bars by subgroup (XGBoost) ----
# ---- (separate files) ----
library(dplyr)
library(ggplot2)
library(purrr)
library(tibble)
library(xgboost)

# Ensure consistent ordering for outcome labels
df$X_case            <- factor(df$X_case,            levels = c("NonHF","HF"))
df_af$X_case         <- factor(df_af$X_case,         levels = c("NonHF","HF"))
df_non_af$X_case     <- factor(df_non_af$X_case,     levels = c("NonHF","HF"))
df_diabetes$X_case   <- factor(df_diabetes$X_case,   levels = c("NonHF","HF"))
df_non_diabetes$X_case <- factor(df_non_diabetes$X_case, levels = c("NonHF","HF"))
df_female$X_case     <- factor(df_female$X_case,     levels = c("NonHF","HF"))
df_male$X_case       <- factor(df_male$X_case,       levels = c("NonHF","HF"))

# Subgroup datasets (as you defined earlier)
subgroups <- list(
  "AF"            = df_af,
  "Non AF"        = df_non_af,
  "Diabetes"      = df_diabetes,
  "Non Diabetes"  = df_non_diabetes,
  "Female"        = df_female,
  "Male"          = df_male
)

# Map for threshold variable names (if you computed threshold_af, threshold_non_af, etc.)
thr_key <- c("AF"="af","Non AF"="non_af","Diabetes"="diabetes",
             "Non Diabetes"="non_diabetes","Female"="female","Male"="male")

get_thr <- function(name, default = 0.01069671) {
  obj <- paste0("threshold_", thr_key[[name]])
  if (exists(obj, inherits = TRUE)) get(obj, inherits = TRUE) else default
}

# Helper: make & (optionally) save one plot
plot_pred_bars <- function(data, subgroup_name, thr = NULL, save_path = NULL) {
  if (nrow(data) == 0) return(NULL)
  
  # Predict probabilities using the full-data XGBoost model
  dmat  <- xgb.DMatrix(data = model.matrix(~ . - 1, data = data %>% select(-X_case)))
  probs <- predict(xgb_fitted, dmat)
  
  # Threshold for classification
  if (is.null(thr)) thr <- get_thr(subgroup_name)
  
  preds <- factor(ifelse(probs > thr, "HF", "NonHF"), levels = c("NonHF","HF"))
  pdat  <- tibble(Predicted = preds) |>
    count(Predicted, name = "n") |>
    mutate(
      percent = n / sum(n),
      label   = paste0(n, " (", sprintf("%.1f%%", percent * 100), ")")
    )
  
  p <- ggplot(pdat, aes(x = Predicted, y = n)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = label), vjust = -0.4, size = 4) +
    labs(
      title = paste0("Predicted outcome — ", subgroup_name, " (XGBoost)"),
      x = "Predicted class", y = "Count"
    ) +
    theme_minimal(base_size = 12)
  
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), showWarnings = FALSE, recursive = TRUE)
    ggsave(save_path, p, width = 8, height = 5, dpi = 300)
  }
  p
}

# Generate & save each subgroup plot (separate files in ./plots/)
plots <- imap(subgroups, ~ {
  fname <- paste0("plots/Predicted_", gsub("[^A-Za-z0-9]+", "", .y), ".png")
  plot_pred_bars(.x, .y, thr = get_thr(.y), save_path = fname)
})

# If you want to display in the R session too, you can print any of them, e.g.:
plots$AF
#plots$Non AF
#plots$Diabetes
#plots$Non Diabetes
#plots$Female
#plots$Male





# Predicted Heart Failure Rates across Subgroups

# Build predicted class counts per subgroup
pred_counts <- imap_dfr(subgroups, function(dat, sg) {
  if (nrow(dat) == 0) return(NULL)
  dmat <- xgb.DMatrix(data = model.matrix(~ . - 1, data = dat %>% select(-X_case)))
  probs <- predict(xgb_fitted, dmat)
  thr   <- get_thr(sg)
  preds <- factor(ifelse(probs > thr, "HF", "NonHF"), levels = c("NonHF","HF"))
  tibble(Subgroup = sg, Predicted = preds)
}) %>%
  count(Subgroup, Predicted, name = "n") %>%
  group_by(Subgroup) %>%
  mutate(percent = n / sum(n),
         label = paste0(n, " (", sprintf("%.1f%%", percent * 100), ")")) %>%
  ungroup()


hf_rate <- pred_counts %>%
  group_by(Subgroup) %>%
  mutate(total = sum(n)) %>%
  ungroup() %>%
  filter(Predicted == "HF") %>%
  transmute(Subgroup, hf_rate = n / total)

ggplot(hf_rate, aes(x = Subgroup, y = hf_rate)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", hf_rate * 100)), vjust = -0.4, size = 3.6) +
  labs(title = "Predicted HF rate by subgroup (XGBoost)",
       x = NULL, y = "Predicted HF (%)") +
  theme_minimal(base_size = 12)

ggsave("Predicted_HF_Rate_by_Subgroup.png", width = 8, height = 5, dpi = 300)


##### EDA 1 #####
#### Data Exploration ####

# 1. Basic structure and summary
str(df)                                # Structure of dataset
summary(df)                            # Descriptive statistics
dim(df)                                # Dimensions of dataset

# 2. Missing values check
colSums(is.na(df))                     # Count of missing values per variable

# 3. Distribution of continuous variables (histograms)
numeric_vars <- df %>%
  select(where(is.numeric))             # Select numeric columns only

par(mfrow = c(2, 2))                    # Plot in grid (adjust if many variables)
for (col in colnames(numeric_vars)) {
  hist(numeric_vars[[col]], 
       main = paste("Distribution of", col), 
       xlab = col, 
       col = "lightblue", 
       border = "white")
}
par(mfrow = c(1, 1))

# 4. Distribution of categorical variables (bar plots)
categorical_vars <- df %>% select(where(is.factor)) %>% colnames()

for (col in categorical_vars) {
  p <- ggplot(df, aes_string(x = col)) +
    geom_bar(fill = "darkseagreen3", color = "black") +
    labs(title = paste("Distribution of", col), x = col, y = "Count") +
    theme_minimal(base_size = 13)
  print(p)
}

# 5. Correlation matrix for numeric features
cor_matrix <- cor(df %>% select(where(is.numeric)), use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "upper",
         tl.cex = 0.8, tl.col = "black", title = "Correlation Heatmap")


# 6. Group-wise summary (subgroup baseline characteristics)
df %>%
  group_by(af) %>%
  summarise(across(where(is.numeric), list(mean = mean, sd = sd), .names = "{col}_{fn}"))

df %>%
  group_by(diabetes) %>%
  summarise(across(where(is.numeric), list(mean = mean, sd = sd), .names = "{col}_{fn}"))

df %>%
  group_by(gender) %>%
  summarise(across(where(is.numeric), list(mean = mean, sd = sd), .names = "{col}_{fn}"))


cor_matrix


### EDA 2 ###
# Outcome distribution with gender split
library(ggplot2)

# Count plot
p1 <- ggplot(df, aes(x = factor(X_case), fill = gender)) +
  geom_bar(position = "dodge", color = "black") +
  labs(title = "Distribution of Heart Failure Status by Gender",
       x = "Heart Failure Status",
       y = "Count",
       fill = "Gender") +
  scale_x_discrete(labels = c("0" = "No Heart Failure", "1" = "Heart Failure")) +
  theme_minimal(base_size = 14)
print(p1)

# Proportional plot (relative percentages by outcome group)
p2 <- ggplot(df, aes(x = factor(X_case), fill = gender)) +
  geom_bar(position = "fill", color = "black") +
  labs(title = "Proportion of Gender within Heart Failure Status",
       x = "Heart Failure Status",
       y = "Proportion",
       fill = "Gender") +
  scale_x_discrete(labels = c("0" = "No Heart Failure", "1" = "Heart Failure")) +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal(base_size = 14)
print(p2)

# Heart Failure status vs AF status

# Count plot
p3 <- ggplot(df, aes(x = factor(X_case), fill = factor(af))) +
  geom_bar(position = "dodge", color = "black") +
  labs(title = "Distribution of Heart Failure Status by AF / Non-AF",
       x = "Heart Failure Status",
       y = "Count",
       fill = "AF Status") +
  scale_x_discrete(labels = c("0" = "No Heart Failure", "1" = "Heart Failure")) +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "tomato"),
                    labels = c("0" = "Non-AF", "1" = "AF")) +
  theme_minimal(base_size = 14)
print(p3)

# Proportional plot
p4 <- ggplot(df, aes(x = factor(X_case), fill = factor(af))) +
  geom_bar(position = "fill", color = "black") +
  labs(title = "Proportion of AF vs Non-AF within Heart Failure Status",
       x = "Heart Failure Status",
       y = "Proportion",
       fill = "AF Status") +
  scale_x_discrete(labels = c("0" = "No Heart Failure", "1" = "Heart Failure")) +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "tomato"),
                    labels = c("0" = "Non-AF", "1" = "AF")) +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal(base_size = 14)
print(p4)

# Heart Failure status vs Diabetes status

# Count plot
p5 <- ggplot(df, aes(x = factor(X_case), fill = factor(diabetes))) +
  geom_bar(position = "dodge", color = "black") +
  labs(title = "Distribution of Heart Failure Status by Diabetes / Non-Diabetes",
       x = "Heart Failure Status",
       y = "Count",
       fill = "Diabetes Status") +
  scale_x_discrete(labels = c("0" = "No Heart Failure", "1" = "Heart Failure")) +
  scale_fill_manual(values = c("0" = "mediumseagreen", "1" = "orange"),
                    labels = c("0" = "Non-Diabetes", "1" = "Diabetes")) +
  theme_minimal(base_size = 14)
print(p5)

# Proportional plot
p6 <- ggplot(df, aes(x = factor(X_case), fill = factor(diabetes))) +
  geom_bar(position = "fill", color = "black") +
  labs(title = "Proportion of Diabetes vs Non-Diabetes within Heart Failure Status",
       x = "Heart Failure Status",
       y = "Proportion",
       fill = "Diabetes Status") +
  scale_x_discrete(labels = c("0" = "No Heart Failure", "1" = "Heart Failure")) +
  scale_fill_manual(values = c("0" = "mediumseagreen", "1" = "orange"),
                    labels = c("0" = "Non-Diabetes", "1" = "Diabetes")) +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal(base_size = 14)
print(p6)


dim(df)


glimpse(df)
str(df)










#### Load SHAPViz Library ####
library(shapviz)

#### Helper Function to Compute SHAP for any XGBoost model + dataset ####
compute_shap <- function(xgb_model, train_data_matrix, predict_data) {
  # train_data_matrix: the matrix used in xgboost() training (model.matrix)
  # predict_data: dataframe to compute SHAP on (numeric outcome, subgroup indicators removed)
  
  # Reorder columns in predict_data to match training data
  predict_data_matrix <- model.matrix(~ . -1, data = predict_data)  # convert to numeric
  predict_data_matrix <- predict_data_matrix[, colnames(train_data_matrix), drop = FALSE]
  
  # Compute SHAP values
  shap_obj <- shapviz(xgb_model, X_pred = predict_data_matrix)
  return(shap_obj)
}

#### Prepare Training Matrices for Each Model ####
# Full data
train_matrix_full <- model.matrix(~ . -1, data = data3_xgb %>% select(-X_case))

# AF subgroup
train_matrix_af <- model.matrix(~ . -1, data = data_af_xgb %>% select(-X_case))
train_matrix_non_af <- model.matrix(~ . -1, data = data_non_af_xgb %>% select(-X_case))

# Diabetes subgroup
train_matrix_diabetes <- model.matrix(~ . -1, data = data_diabetes_xgb %>% select(-X_case))
train_matrix_non_diabetes <- model.matrix(~ . -1, data = data_non_diabetes_xgb %>% select(-X_case))

# Gender subgroup
train_matrix_female <- model.matrix(~ . -1, data = data_female_xgb %>% select(-X_case))
train_matrix_male <- model.matrix(~ . -1, data = data_male_xgb %>% select(-X_case))

#### Compute SHAP Values ####
shap_full <- compute_shap(xgb_fitted, train_matrix_full, df %>% select(-X_case))
shap_af <- compute_shap(xgb_fitted_af, train_matrix_af, df_af_xgb %>% select(-X_case))
shap_non_af <- compute_shap(xgb_fitted_non_af, train_matrix_non_af, df_non_af_xgb %>% select(-X_case))
shap_diabetes <- compute_shap(xgb_fitted_diabetes, train_matrix_diabetes, df_diabetes_xgb %>% select(-X_case))
shap_non_diabetes <- compute_shap(xgb_fitted_non_diabetes, train_matrix_non_diabetes, df_non_diabetes_xgb %>% select(-X_case))
shap_female <- compute_shap(xgb_fitted_female, train_matrix_female, df_female_xgb %>% select(-X_case))
shap_male <- compute_shap(xgb_fitted_male, train_matrix_male, df_male_xgb %>% select(-X_case))

#### SHAP Summary Plots ####
# Beeswarm for full data
sv_importance(shap_full) %>% plot(beeswarm = TRUE)

# Dependence plot for top variable (age_index)
plot(sv_dependence(shap_full, "age_index"))

# Force plot for first 20 observations
plot(sv_force(shap_full, row_ids = 1:20))

# Waterfall plot for first observation
plot(sv_waterfall(shap_full, row_id = 1))



sv_importance(shap_af) %>% plot(beeswarm = TRUE)
sv_importance(shap_non_af) %>% plot(beeswarm = TRUE)
sv_importance(shap_diabetes) %>% plot(beeswarm = TRUE)
sv_importance(shap_non_diabetes) %>% plot(beeswarm = TRUE)
sv_importance(shap_female) %>% plot(beeswarm = TRUE)
sv_importance(shap_male) %>% plot(beeswarm = TRUE)


shap_full

