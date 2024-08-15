# Load necessary libraries
library(ggplot2)
library(dplyr)
library(MASS)
library(rpart)
library(rpart.plot)
library(VIM)
# Load necessary libraries
library(randomForest)
# Load necessary libraries
library(smbinning)
# Load necessary libraries
library(caret)

# Load data
df_1 <- read.csv("credit_risk_dataset_full.csv")

# Ensure BAD is a factor for classification
df_1$loan_status <- as.factor(df_1$loan_status)

# Define the grid of hyperparameters to search
mtry_values <- c(2, 3, 4, 5, 6, 7)
ntree_values <- c(600, 700, 800)

# Initialize variables to store best results
best_accuracy <- 0
best_rf_model <- NULL
best_params <- list()

# Define a function to calculate accuracy
calculate_accuracy <- function(model, data) {
  predictions <- predict(model, data)
  confusion_matrix <- table(predictions, data$loan_status)
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  return(accuracy)
}

# Perform grid search manually
set.seed(123)
for (mtry in mtry_values) {
  for (ntree in ntree_values) {
    model <- randomForest(loan_status ~ person_age+person_income
                          +person_home_ownership+person_emp_length
                          +loan_intent+loan_grade+loan_amnt
                          +loan_int_rate+loan_percent_income+
                            cb_person_default_on_file+
                            cb_person_cred_hist_length
                          , data = df_1, mtry = mtry, ntree = ntree)
    accuracy <- calculate_accuracy(model, df_1)
    
    if (accuracy > best_accuracy) {
      best_accuracy <- accuracy
      best_rf_model <- model
      best_params <- list(mtry = mtry, ntree = ntree)
    }
  }
}

# Print the best parameters and the corresponding accuracy
print(best_params)
print(best_accuracy)

# Print the best random forest model
print(best_rf_model)

# Calculate variable importance
importance_values <- importance(best_rf_model, type=2, scale=TRUE)

# Print variable importance
print(importance_values)

# Plot variable importance
varImpPlot(best_rf_model)