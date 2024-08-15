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
# Load necessary libraries
library(writexl)
library(openxlsx)

# Load data
df <- read.csv("credit_risk_dataset.csv")

# Delete outliers
df$person_age[df$person_age > 100] <- NA
df$person_emp_length[df$person_emp_length > 50] <- NA

# Check for missing values in each column
missing_values <- colSums(is.na(df))

# Print the number of missing values in each column
print(missing_values)

# Use KNN to impute missing values (assuming df is your data frame)
df_1 <- kNN(df, k = 5)

# kNN function will generate new columns, remove new columns to keep original columns
df_1 <- df_1[, 1:ncol(df)]

# Check if there are any missing values left
missing_values_after <- colSums(is.na(df_1))

# Print the number of missing values in each column after imputation
print(missing_values_after)

# Check column names
columns <- colnames(df_1)
print(columns)

attach(df_1)

write.csv(df_1, file = "credit_risk_dataset_full.csv")


set.seed(123)
trainIndex <- createDataPartition(df_1$loan_status,
                                  p = 0.7, list = FALSE)

# Create training and test sets
df_train <- df_1[trainIndex, ]
df_test <- df_1[-trainIndex, ]

# Check the proportion of loan_status in the training and test sets
prop_train <- prop.table(table(df_train$loan_status))
prop_test <- prop.table(table(df_test$loan_status))

# Print proportions
print("Training set loan_status proportion:")
print(prop_train)

print("Test set loan_status proportion:")
print(prop_test)

# Build logistic regression model
logit_model <- glm(loan_status ~ .,
                   data = df_train, family = "binomial")

df_train$pred_logit <- predict(logit_model, type = "response")

df_test$pred_logit <- predict(logit_model,
                              newdata = df_test, type = "response")

# Create decision tree model
tree_model_train <- rpart(loan_status ~ pred_logit,
                    data = df_train, method = "class",
                    control = rpart.control
                    (minsplit = 300, minbucket = 300, 
                      maxdepth = 3, cp = 0.0001))

# Plot the modified decision tree
rpart.plot(tree_model_train, type = 4, extra = 101, 
           under = TRUE, cex = 0.8, box.palette = "auto")

# Print rules
print(rpart.rules(tree_model_train))

# Predict probabilities for each sample using decision tree
pred_prob_train <- predict(tree_model_train, type = "prob")[, 2]

# Add predicted probabilities to the data frame
df_train <- df_train %>%
  mutate(pred_prob_train = pred_prob_train)

# Predict classifications for each sample using decision tree
pred_tree_train <- predict(tree_model_train, type = "class")

# Create confusion matrix
conf_matrix_train <- table(Predicted = pred_tree_train, 
                           Actual = df_train$loan_status)

# Print confusion matrix
print(conf_matrix_train)

# Predict probabilities for each sample in the test set using the decision tree
pred_prob_test <- predict(tree_model_train, newdata = df_test, type = "prob")[, 2]

# Add predicted probabilities to the test data frame
df_test <- df_test %>%
  mutate(pred_prob_test = pred_prob_test)

# Predict classifications for each sample in the test set using the decision tree
pred_tree_test <- predict(tree_model_train, newdata = df_test, type = "class")

# Create confusion matrix for the test set
conf_matrix_test <- table(Predicted = pred_tree_test, Actual = df_test$loan_status)

# Print confusion matrix for the test set
print(conf_matrix_test)





# Sort training data by logistic model predictions in descending order
df_train <- df_train %>%
  arrange(desc(pred_logit))

# Calculate cumulative bad samples and total bad samples for the training data
df_train <- df_train %>%
  mutate(cum_bad_logit_train = cumsum(loan_status),
         total_bad_logit_train = sum(loan_status))

# Calculate CAP curve data for logistic model on the training data
df_train <- df_train %>%
  mutate(cum_perc_bad_logit_train = 
           cum_bad_logit_train / total_bad_logit_train,
         cum_perc_total_logit_train = row_number() / n())

# Sort training data by decision tree predicted probabilities in descending order
df_train <- df_train %>%
  arrange(desc(pred_prob_train))

# Group by rounded predicted probabilities and calculate counts and bad counts in the training data
df_tree_grouped_train <- df_train %>%
  mutate(pred_prob_train = round(pred_prob_train, 3)) %>%
  group_by(pred_prob_train) %>%
  summarise(count_train = n(),
            bad_count_train = sum(loan_status)) %>%
  ungroup() %>%
  arrange(desc(pred_prob_train))




# Define interval boundaries
breaks <- c(-Inf, 0.12, 0.35, 0.68, Inf)

# Group df_train by intervals
df_tree_grouped_train <- df_train %>%
  mutate(group_train = cut(pred_logit, breaks = breaks, 
                           include.lowest = TRUE)) %>%
  group_by(group_train) %>%
  summarise(
    count_train = n(),
    bad_count_train = sum(loan_status),
    pred_prob_train = round(mean(loan_status), 3)
  )

# Sort df_tree_grouped_train by pred_prob_train in descending order
df_tree_grouped_train <- df_tree_grouped_train %>%
  arrange(desc(pred_prob_train))

# View sorted df_tree_grouped_train
print(df_tree_grouped_train)

# Add initial point (0, 0) for training data
initial_point_train <- data.frame(pred_prob_train = 0, 
                                  count_train = 0, bad_count_train = 0)
df_tree_grouped_train_0 <- bind_rows(initial_point_train, 
                                     df_tree_grouped_train)

# Calculate cumulative bad samples and total samples for training data
df_tree_grouped_train_0 <- df_tree_grouped_train_0 %>%
  mutate(cum_bad_tree_train = cumsum(bad_count_train),
         cum_total_tree_train = cumsum(count_train),
         total_bad_tree_train = sum(df_train$loan_status),
         total_count_tree_train = nrow(df_train))

# Calculate CAP curve data for training data
df_tree_grouped_train_0 <- df_tree_grouped_train_0 %>%
  mutate(cum_perc_bad_tree_train = round(cum_bad_tree_train /
                                           total_bad_tree_train, 3),
         cum_perc_total_tree_train = round(cum_total_tree_train /
                                             total_count_tree_train, 3))

# Calculate perfect model data for training data
perfect_model_train <- df_train %>%
  arrange(desc(loan_status)) %>%
  mutate(cum_bad_train = cumsum(loan_status),
         total_bad_train = sum(loan_status),
         cum_perc_bad_train = cum_bad_train / total_bad_train,
         cum_perc_total_train = row_number() / n())


# Define the function to calculate Accuracy Ratio (AR)
calculate_ar <- function(df, cum_perc_total_col, cum_perc_bad_col, total_bad, total_count) {
  # Calculate AUC for the actual model
  auc_actual <- sum((df[[cum_perc_total_col]][-1] - df[[cum_perc_total_col]][-nrow(df)]) * 
                      (df[[cum_perc_bad_col]][-1] + df[[cum_perc_bad_col]][-nrow(df)]) / 2)
  
  # Calculate AUC for the perfect model
  auc_perfect <- 1 - (total_bad / total_count) * 0.5
  
  # Calculate AUC for the random model
  auc_random <- 0.5
  
  # Calculate Accuracy Ratio
  ar <- (auc_actual - auc_random) / (auc_perfect - auc_random)
  
  return(ar)
}

# Define the total bad and total count as single values
total_bad_logit_train <- sum(df_train$loan_status)
total_bad_tree_train <- sum(df_train$loan_status)
total_count_logit_train <- nrow(df_train)
total_count_tree_train <- nrow(df_train)

# Calculate AR for logistic model
ar_logit_train <- calculate_ar(df_train, 
                               "cum_perc_total_logit_train", 
                               "cum_perc_bad_logit_train", 
                               total_bad_logit_train,
                               total_count_logit_train)

# Calculate AR for decision tree model
ar_tree_train <- calculate_ar(df_tree_grouped_train_0,
                              "cum_perc_total_tree_train", 
                              "cum_perc_bad_tree_train", 
                              total_bad_tree_train,
                              total_count_tree_train)


# Plot CAP curves for logistic model and decision tree model on training data, with labels
ggplot() +
  geom_line(data = df_train, aes(x = cum_perc_total_logit_train, y = cum_perc_bad_logit_train), color = "blue", linewidth = 1) +
  geom_line(data = df_tree_grouped_train_0, aes(x = cum_perc_total_tree_train, y = cum_perc_bad_tree_train), color = "orange", linewidth = 1) +
  geom_line(data = perfect_model_train, aes(x = cum_perc_total_train, y = cum_perc_bad_train), color = "green", linetype = "dashed", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dotted", linewidth = 1) +
  geom_point(data = df_tree_grouped_train_0, aes(x = cum_perc_total_tree_train, y = cum_perc_bad_tree_train), color = "orange", size = 3) +
  geom_segment(data = df_tree_grouped_train_0, aes(x = cum_perc_total_tree_train, xend = cum_perc_total_tree_train, y = 0, yend = cum_perc_bad_tree_train), color = "orange", linetype = "dashed") +
  annotate("text", x = 0.6, y = 0.4, label = paste("Logit Model (AR:", round(ar_logit_train, 3), ")"), color = "blue", hjust = 0) +
  annotate("text", x = 0.6, y = 0.5, label = paste("Tree Model (AR:", round(ar_tree_train, 3), ")"), color = "orange", hjust = 0) +
  annotate("text", x = 0.6, y = 0.6, label = "Perfect Model", color = "green", hjust = 0) +
  annotate("text", x = 0.6, y = 0.3, label = "Random Model", color = "red", hjust = 0) +
  annotate("text", x = df_tree_grouped_train_0$cum_perc_total_tree_train[1], y = -0.02, label = round(df_tree_grouped_train_0$cum_perc_total_tree_train[1], 2), color = "orange", vjust = 1) +
  annotate("text", x = df_tree_grouped_train_0$cum_perc_total_tree_train[2], y = -0.02, label = round(df_tree_grouped_train_0$cum_perc_total_tree_train[2], 2), color = "orange", vjust = 1) +
  annotate("text", x = df_tree_grouped_train_0$cum_perc_total_tree_train[3], y = -0.02, label = round(df_tree_grouped_train_0$cum_perc_total_tree_train[3], 2), color = "orange", vjust = 1) +
  annotate("text", x = df_tree_grouped_train_0$cum_perc_total_tree_train[4], y = -0.02, label = round(df_tree_grouped_train_0$cum_perc_total_tree_train[4], 2), color = "orange", vjust = 1) +
  annotate("text", x = df_tree_grouped_train_0$cum_perc_total_tree_train[5], y = -0.02, label = round(df_tree_grouped_train_0$cum_perc_total_tree_train[5], 2), color = "orange", vjust = 1) +
  labs(title = "CAP Curve for logit_model and tree_model on Training Data",
       x = "Cumulative Percentage of Sorted Samples",
       y = "Cumulative Percentage of Bad Captured") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Remove the initial point from training data
df_tree_segmentation_train <- df_tree_grouped_train_0[-1, ]

# Calculate actual default rate for each segment in the training data
df_tree_segmentation_train <- df_tree_segmentation_train %>%
  mutate(actual_default_rate_train = bad_count_train / count_train)

# Plot actual default rate by predicted probability group in the training data
ggplot(df_tree_segmentation_train, aes(x = factor(pred_prob_train),
                                       y = actual_default_rate_train)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Actual Default Rate by Predicted Probability on Training Data",
       x = "Predicted Probability Group",
       y = "Actual Default Rate") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# Test Set Processing

# Sort test data by logistic model predictions in descending order
df_test <- df_test %>%
  arrange(desc(pred_logit))

# Calculate cumulative bad samples and total bad samples for the test data
df_test <- df_test %>%
  mutate(cum_bad_logit_test = cumsum(loan_status),
         total_bad_logit_test = sum(loan_status))

# Calculate CAP curve data for logistic model on the test data
df_test <- df_test %>%
  mutate(cum_perc_bad_logit_test = 
           cum_bad_logit_test / total_bad_logit_test,
         cum_perc_total_logit_test = row_number() / n())

# Sort test data by decision tree predicted probabilities in descending order
df_test <- df_test %>%
  arrange(desc(pred_prob_test))

# Group by rounded predicted probabilities and calculate counts and bad counts in the test data
df_tree_grouped_test <- df_test %>%
  mutate(pred_prob_test = round(pred_prob_test, 3)) %>%
  group_by(pred_prob_test) %>%
  summarise(count_test = n(),
            bad_count_test = sum(loan_status)) %>%
  ungroup() %>%
  arrange(desc(pred_prob_test))

# Define interval boundaries
breaks <- c(-Inf, 0.12, 0.35, 0.68, Inf)

# Group df_test by intervals
df_tree_grouped_test <- df_test %>%
  mutate(group_test = cut(pred_logit, breaks = breaks, 
                          include.lowest = TRUE)) %>%
  group_by(group_test) %>%
  summarise(
    count_test = n(),
    bad_count_test = sum(loan_status),
    pred_prob_test = round(mean(loan_status), 3)
  )

# Sort df_tree_grouped_test by pred_prob_test in descending order
df_tree_grouped_test <- df_tree_grouped_test %>%
  arrange(desc(pred_prob_test))

# View sorted df_tree_grouped_test
print(df_tree_grouped_test)

# Add initial point (0, 0) for test data
initial_point_test <- data.frame(pred_prob_test = 0, 
                                 count_test = 0, bad_count_test = 0)
df_tree_grouped_test_0 <- bind_rows(initial_point_test, 
                                    df_tree_grouped_test)

# Calculate cumulative bad samples and total samples for test data
df_tree_grouped_test_0 <- df_tree_grouped_test_0 %>%
  mutate(cum_bad_tree_test = cumsum(bad_count_test),
         cum_total_tree_test = cumsum(count_test),
         total_bad_tree_test = sum(df_test$loan_status),
         total_count_tree_test = nrow(df_test))

# Calculate CAP curve data for test data
df_tree_grouped_test_0 <- df_tree_grouped_test_0 %>%
  mutate(cum_perc_bad_tree_test = round(cum_bad_tree_test /
                                          total_bad_tree_test, 3),
         cum_perc_total_tree_test = round(cum_total_tree_test /
                                            total_count_tree_test, 3))

# Calculate perfect model data for test data
perfect_model_test <- df_test %>%
  arrange(desc(loan_status)) %>%
  mutate(cum_bad_test = cumsum(loan_status),
         total_bad_test = sum(loan_status),
         cum_perc_bad_test = cum_bad_test / total_bad_test,
         cum_perc_total_test = row_number() / n())

# Define the total bad and total count as single values for test data
total_bad_logit_test <- sum(df_test$loan_status)
total_bad_tree_test <- sum(df_test$loan_status)
total_count_logit_test <- nrow(df_test)
total_count_tree_test <- nrow(df_test)

# Calculate AR for logistic model on test data
ar_logit_test <- calculate_ar(df_test, 
                              "cum_perc_total_logit_test", 
                              "cum_perc_bad_logit_test", 
                              total_bad_logit_test,
                              total_count_logit_test)

# Calculate AR for decision tree model on test data
ar_tree_test <- calculate_ar(df_tree_grouped_test_0,
                             "cum_perc_total_tree_test", 
                             "cum_perc_bad_tree_test", 
                             total_bad_tree_test,
                             total_count_tree_test)

# Plot CAP curves for logistic model and decision tree model on test data, with labels
ggplot() +
  geom_line(data = df_test, aes(x = cum_perc_total_logit_test, y = cum_perc_bad_logit_test), color = "blue", linewidth = 1) +
  geom_line(data = df_tree_grouped_test_0, aes(x = cum_perc_total_tree_test, y = cum_perc_bad_tree_test), color = "orange", linewidth = 1) +
  geom_line(data = perfect_model_test, aes(x = cum_perc_total_test, y = cum_perc_bad_test), color = "green", linetype = "dashed", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dotted", linewidth = 1) +
  geom_point(data = df_tree_grouped_test_0, aes(x = cum_perc_total_tree_test, y = cum_perc_bad_tree_test), color = "orange", size = 3) +
  geom_segment(data = df_tree_grouped_test_0, aes(x = cum_perc_total_tree_test, xend = cum_perc_total_tree_test, y = 0, yend = cum_perc_bad_tree_test), color = "orange", linetype = "dashed") +
  annotate("text", x = 0.6, y = 0.4, label = paste("Logit Model (AR:", round(ar_logit_test, 3), ")"), color = "blue", hjust = 0) +
  annotate("text", x = 0.6, y = 0.5, label = paste("Tree Model (AR:", round(ar_tree_test, 3), ")"), color = "orange", hjust = 0) +
  annotate("text", x = 0.6, y = 0.6, label = "Perfect Model", color = "green", hjust = 0) +
  annotate("text", x = 0.6, y = 0.3, label = "Random Model", color = "red", hjust = 0) +
  labs(title = "CAP Curve for logit_model and tree_model on Test Data",
       x = "Cumulative Percentage of Sorted Samples",
       y = "Cumulative Percentage of Bad Captured") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Remove the initial point from test data
df_tree_segmentation_test <- df_tree_grouped_test_0[-1, ]

# Calculate actual default rate for each segment in the test data
df_tree_segmentation_test <- df_tree_segmentation_test %>%
  mutate(actual_default_rate_test = bad_count_test / count_test)

# Plot actual default rate by predicted probability group in the test data
ggplot(df_tree_segmentation_test, aes(x = factor(pred_prob_test),
                                      y = actual_default_rate_test)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Actual Default Rate by Predicted Probability on Test Data",
       x = "Predicted Probability Group",
       y = "Actual Default Rate") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))



# Train Set T-Tests

# Extract BAD data for each group in train set with new thresholds
group1_train <- df_train %>% filter(pred_logit > -Inf & pred_logit <= 0.12) %>% pull(loan_status)
group2_train <- df_train %>% filter(pred_logit > 0.12 & pred_logit <= 0.35) %>% pull(loan_status)
group3_train <- df_train %>% filter(pred_logit > 0.35 & pred_logit <= 0.68) %>% pull(loan_status)
group4_train <- df_train %>% filter(pred_logit > 0.68 & pred_logit <= Inf) %>% pull(loan_status)

# Perform t-tests for train set
t_test_1_2_train <- t.test(group1_train, group2_train)
t_test_2_3_train <- t.test(group2_train, group3_train)
t_test_3_4_train <- t.test(group3_train, group4_train)

# Store t-test results for train set
t_tests_train <- list(
  t_test_1_2_train = t.test(group1_train, group2_train),
  t_test_2_3_train = t.test(group2_train, group3_train),
  t_test_3_4_train = t.test(group3_train, group4_train)
)

# Create a data frame to store t-test results for train set
df_t_test_train <- data.frame(
  group1 = c("group1", "group2", "group3"),
  group2 = c("group2", "group3", "group4"),
  t_statistic = sapply(t_tests_train, function(x) x$statistic),
  p_value = sapply(t_tests_train, function(x) x$p.value),
  conf_low = sapply(t_tests_train, function(x) x$conf.int[1]),
  conf_high = sapply(t_tests_train, function(x) x$conf.int[2]),
  mean_group1 = sapply(t_tests_train, function(x) x$estimate[1]),
  mean_group2 = sapply(t_tests_train, function(x) x$estimate[2])
)

# View df_t_test_train
print(df_t_test_train)

# Test Set T-Tests

# Extract BAD data for each group in test set with new thresholds
group1_test <- df_test %>% filter(pred_logit > -Inf & pred_logit <= 0.12) %>% pull(loan_status)
group2_test <- df_test %>% filter(pred_logit > 0.12 & pred_logit <= 0.35) %>% pull(loan_status)
group3_test <- df_test %>% filter(pred_logit > 0.35 & pred_logit <= 0.68) %>% pull(loan_status)
group4_test <- df_test %>% filter(pred_logit > 0.68 & pred_logit <= Inf) %>% pull(loan_status)

# Perform t-tests for test set
t_test_1_2_test <- t.test(group1_test, group2_test)
t_test_2_3_test <- t.test(group2_test, group3_test)
t_test_3_4_test <- t.test(group3_test, group4_test)

# Store t-test results for test set
t_tests_test <- list(
  t_test_1_2_test = t.test(group1_test, group2_test),
  t_test_2_3_test = t.test(group2_test, group3_test),
  t_test_3_4_test = t.test(group3_test, group4_test)
)

# Create a data frame to store t-test results for test set
df_t_test_test <- data.frame(
  group1 = c("group1", "group2", "group3"),
  group2 = c("group2", "group3", "group4"),
  t_statistic = sapply(t_tests_test, function(x) x$statistic),
  p_value = sapply(t_tests_test, function(x) x$p.value),
  conf_low = sapply(t_tests_test, function(x) x$conf.int[1]),
  conf_high = sapply(t_tests_test, function(x) x$conf.int[2]),
  mean_group1 = sapply(t_tests_test, function(x) x$estimate[1]),
  mean_group2 = sapply(t_tests_test, function(x) x$estimate[2])
)

# View df_t_test_test
print(df_t_test_test)












# Ensure BAD is a factor for classification
df$loan_status <- as.factor(df$loan_status)

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
                          , data = df, mtry = mtry, ntree = ntree)
    accuracy <- calculate_accuracy(model, df)
    
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