# Load necessary libraries
library(dplyr)
library(rpart)
library(rpart.plot)
library(smbinning)

# Read data
df_1 <- read.csv("credit_risk_dataset_full.csv")

# Ensure loan status is a factor for classification
df_1$loan_status <- as.factor(df_1$loan_status)
df_1$loan_grade <- as.factor(df_1$loan_grade)

set.seed(123)
trainIndex <- createDataPartition(df_1$loan_status,
                                  p = 0.7, list = FALSE)

# Create training and test sets
df_train_WOE <- df_1[trainIndex, ]
df_test_WOE <- df_1[-trainIndex, ]

# Assuming the top 5 variables from your importance plot are:
top_vars <- c("loan_percent_income", "person_income", 
              "loan_int_rate", "loan_grade", "person_home_ownership")

# Initialize a list to store the split points for each variable
split_points_list <- list()

# Loop through each variable and build a decision tree model
for (var in top_vars) {
  
  # Print current variable being processed
  print(paste("Processing variable:", var))
  
  # Create a formula for the decision tree
  formula <- as.formula(paste("loan_status ~", var))
  
  # Build the decision tree model
  tree_model <- rpart(formula, data = df_train_WOE, method = "class",
                      control = rpart.control(minsplit = 100, minbucket = 100, 
                                              maxdepth = 4, cp = 0.0001))
  
  # Manually specify the split points for 'loan_grade'
  if (var == "loan_grade") {
    split_points <- list(left = c("A", "B", "C"), right = c("D", "E", "F", "G"))
    split_points_list[[var]] <- split_points
    print(paste("Manually specified split points for", var, ":", 
                paste(split_points$left, collapse = ", "), 
                "and", paste(split_points$right, collapse = ", ")))
  } else {
    # For numeric variables, extract the split points as before
    splits <- unique(tree_model$splits[, "index"])
    split_points_list[[var]] <- splits
    print(paste("Split points for", var, ":", paste(splits, collapse = ", ")))
  }
  
  # Optional: Plot the decision tree
  rpart.plot(tree_model, main = paste("Decision Tree for", var),
             type = 4, extra = 101, 
             under = TRUE, cex = 0.8, box.palette = "auto")
}
# The split_points_list now contains the split points for each variable

cuts_person_income <- split_points_list[["person_income"]]
cuts_loan_int_rate <- split_points_list[["loan_int_rate"]]

# Create a copy of the original dataset for binning
df_binned_train <- df_train_WOE

# 1. Manually bin loan_percent_income using specified breaks and calculate probabilities
df_binned_train <- df_binned_train %>%
  mutate(loan_percent_income_train_bin = cut(loan_percent_income, 
                                             breaks = 
                                               c(-Inf, 0.15, 0.23, 0.305, 0.45, Inf),
                                             include.lowest = TRUE)) %>%
  group_by(loan_percent_income_train_bin) %>%
  mutate(loan_percent_income_train_prob = mean(loan_status == 1, na.rm = TRUE)) %>%
  ungroup()

# 2. Manually bin person_income using provided split points and calculate probabilities
df_binned_train <- df_binned_train %>%
  mutate(person_income_train_bin = cut(person_income, 
                                       breaks = c(-Inf, 19999, 28999,
                                                  29100, 34999, 39999,
                                                  69999, 89999, Inf),
                                       include.lowest = TRUE)) %>%
  group_by(person_income_train_bin) %>%
  mutate(person_income_train_prob = mean(loan_status == 1, na.rm = TRUE)) %>%
  ungroup()

# 3. Manually bin loan_int_rate using provided split points and calculate probabilities
df_binned_train <- df_binned_train %>%
  mutate(loan_int_rate_train_bin = cut(loan_int_rate, 
                                       breaks = c(-Inf,7, 10, 13, 14, 14.37, 14.6,
                                                  14.82, 15.96, Inf),
                                       include.lowest = TRUE)) %>%
  group_by(loan_int_rate_train_bin) %>%
  mutate(loan_int_rate_train_prob = mean(loan_status == 1, na.rm = TRUE)) %>%
  ungroup()

# 4. For loan_grade (categorical variable), bin each category separately and calculate probabilities
df_binned_train <- df_binned_train %>%
  mutate(loan_grade_train_prob = case_when(
    loan_grade == "A" ~ mean(loan_status[loan_grade == "A"] == 1, na.rm = TRUE),
    loan_grade == "B" ~ mean(loan_status[loan_grade == "B"] == 1, na.rm = TRUE),
    loan_grade == "C" ~ mean(loan_status[loan_grade == "C"] == 1, na.rm = TRUE),
    loan_grade == "D" ~ mean(loan_status[loan_grade == "D"] == 1, na.rm = TRUE),
    loan_grade == "E" ~ mean(loan_status[loan_grade == "E"] == 1, na.rm = TRUE),
    loan_grade == "F" ~ mean(loan_status[loan_grade == "F"] == 1, na.rm = TRUE),
    loan_grade == "G" ~ mean(loan_status[loan_grade == "G"] == 1, na.rm = TRUE)
  ))

# Define the variables to summarize
bin_vars <- c("loan_percent_income_train_bin", "person_income_train_bin", 
              "loan_int_rate_train_bin", "loan_grade")

# Define a list to store the summary results
summary_results <- list()

# Loop through each variable and calculate the summaries
for (var in bin_vars) {
  summary_df <- df_binned_train %>%
    group_by_at(var) %>%
    summarise(
      count = n(),
      loan_status_1_count = sum(loan_status == 1, na.rm = TRUE),
      prob = mean(loan_status == 1, na.rm = TRUE)
    )
  
  # Store the result in the summary_results list
  summary_results[[var]] <- summary_df
  
  # Print the summary for each variable
  print(paste("Summary for", var))
  print(summary_df)
}

# Define the WOE transformation function
woe_transform <- function(prob_column) {
  return(log((prob_column+0.0001) / (1+0.0001 - prob_column)))
}

# Apply the WOE transformation to the four columns and store the results
df_binned_train$loan_percent_income_train_WOE <- woe_transform(df_binned_train$loan_percent_income_train_prob)
df_binned_train$person_income_train_WOE <- woe_transform(df_binned_train$person_income_train_prob)
df_binned_train$loan_int_rate_train_WOE <- woe_transform(df_binned_train$loan_int_rate_train_prob)
df_binned_train$loan_grade_train_WOE <- woe_transform(df_binned_train$loan_grade_train_prob)

# logistic model
logit_model_binned_train <- glm(loan_status ~ 
                                  loan_grade_train_WOE + 
                                  loan_int_rate_train_WOE +
                                  loan_percent_income_train_WOE+
                                  person_income_train_WOE +
                                  loan_grade_train_WOE*loan_int_rate_train_WOE+
                                  loan_grade_train_WOE:loan_percent_income_train_WOE
                                  ,data = df_binned_train, family = "binomial")

# probabilities for logistic model
df_binned_train$predicted_prob <- predict(logit_model_binned_train, 
                                          newdata = df_binned_train, 
                                          type = "response")

df_binned_train$loan_status <- as.numeric(
  as.character(df_binned_train$loan_status))

# Sort df_binned_train by predicted_prob in descending order and group by predicted_prob
df_binned_train_grouped <- df_binned_train %>%
  arrange(desc(predicted_prob)) %>%
  group_by(predicted_prob) %>%
  summarise(count = n(),
            bad_count = sum(loan_status))

# Sort df_tree_grouped_train by pred_prob_train in descending order
df_binned_train_grouped <- df_binned_train_grouped %>%
  arrange(desc(predicted_prob))

# Add initial point (0, 0) for CAP curve
initial_point <- data.frame(predicted_prob = 0, count = 0, bad_count = 0)
df_binned_train_grouped <- bind_rows(initial_point, df_binned_train_grouped)

# Calculate cumulative bad samples and total samples
df_binned_train_grouped <- df_binned_train_grouped %>%
  mutate(cum_bad = cumsum(bad_count),
         cum_total = cumsum(count),
         total_bad = sum(df_binned_train$loan_status),
         total_count = sum(df_binned_train_grouped$count))

# Calculate CAP curve data
df_binned_train_grouped <- df_binned_train_grouped %>%
  mutate(cum_perc_bad = round(cum_bad / total_bad, 3),
         cum_perc_total = round(cum_total / total_count, 3))

# Calculate perfect model data
perfect_model <- df_binned_train %>%
  arrange(desc(loan_status)) %>%
  mutate(cum_bad = cumsum(loan_status),
         total_bad = sum(loan_status),
         cum_perc_bad = cum_bad / total_bad,
         cum_perc_total = row_number() / n())

# Step 3: Define the function to calculate Accuracy Ratio (AR)
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

# Step 4: Calculate AR for the logistic model
ar_logit_train <- calculate_ar(df_binned_train_grouped, 
                               "cum_perc_total", 
                               "cum_perc_bad", 
                               df_binned_train_grouped$total_bad[1],
                               df_binned_train_grouped$total_count[1])

# Create a copy of the original test dataset for binning
df_binned_test <- df_test_WOE

# 1. Manually bin loan_percent_income using specified breaks and calculate probabilities
df_binned_test <- df_binned_test %>%
  mutate(loan_percent_income_test_bin = cut(loan_percent_income, 
                                            breaks = 
                                              c(-Inf, 0.15, 0.23, 0.305, 0.45, Inf),
                                            include.lowest = TRUE)) %>%
  group_by(loan_percent_income_test_bin) %>%
  mutate(loan_percent_income_test_prob = mean(loan_status == 1, na.rm = TRUE)) %>%
  ungroup()

# 2. Manually bin person_income using provided split points and calculate probabilities
df_binned_test <- df_binned_test %>%
  mutate(person_income_test_bin = cut(person_income, 
                                      breaks = c(-Inf, 19999, 28999,
                                                 29100, 34999, 39999,
                                                 69999, 89999, Inf),
                                      include.lowest = TRUE)) %>%
  group_by(person_income_test_bin) %>%
  mutate(person_income_test_prob = mean(loan_status == 1, na.rm = TRUE)) %>%
  ungroup()

# 3. Manually bin loan_int_rate using provided split points and calculate probabilities
df_binned_test <- df_binned_test %>%
  mutate(loan_int_rate_test_bin = cut(loan_int_rate, 
                                      breaks = c(-Inf, 7, 10, 13, 14, 14.37, 14.6,
                                                 14.82, 15.96, Inf),
                                      include.lowest = TRUE)) %>%
  group_by(loan_int_rate_test_bin) %>%
  mutate(loan_int_rate_test_prob = mean(loan_status == 1, na.rm = TRUE)) %>%
  ungroup()

# 4. For loan_grade (categorical variable), bin each category separately and calculate probabilities
df_binned_test <- df_binned_test %>%
  mutate(loan_grade_test_prob = case_when(
    loan_grade == "A" ~ mean(loan_status[loan_grade == "A"] == 1, na.rm = TRUE),
    loan_grade == "B" ~ mean(loan_status[loan_grade == "B"] == 1, na.rm = TRUE),
    loan_grade == "C" ~ mean(loan_status[loan_grade == "C"] == 1, na.rm = TRUE),
    loan_grade == "D" ~ mean(loan_status[loan_grade == "D"] == 1, na.rm = TRUE),
    loan_grade == "E" ~ mean(loan_status[loan_grade == "E"] == 1, na.rm = TRUE),
    loan_grade == "F" ~ mean(loan_status[loan_grade == "F"] == 1, na.rm = TRUE),
    loan_grade == "G" ~ mean(loan_status[loan_grade == "G"] == 1, na.rm = TRUE)
  ))

# Define the variables to summarize
bin_vars <- c("loan_percent_income_test_bin", "person_income_test_bin", 
              "loan_int_rate_test_bin", "loan_grade")

# Define a list to store the summary results
summary_results <- list()

# Loop through each variable and calculate the summaries
for (var in bin_vars) {
  summary_df <- df_binned_test %>%
    group_by_at(var) %>%
    summarise(
      count = n(),
      loan_status_1_count = sum(loan_status == 1, na.rm = TRUE),
      prob = mean(loan_status == 1, na.rm = TRUE)
    )
  
  # Store the result in the summary_results list
  summary_results[[var]] <- summary_df
  
  # Print the summary for each variable
  print(paste("Summary for", var))
  print(summary_df)
}

# Apply the WOE transformation to the four columns and store the results
df_binned_test$loan_percent_income_test_WOE <- woe_transform(df_binned_test$loan_percent_income_test_prob)
df_binned_test$person_income_test_WOE <- woe_transform(df_binned_test$person_income_test_prob)
df_binned_test$loan_int_rate_test_WOE <- woe_transform(df_binned_test$loan_int_rate_test_prob)
df_binned_test$loan_grade_test_WOE <- woe_transform(df_binned_test$loan_grade_test_prob)

# logistic model
logit_model_binned_test <- glm(loan_status ~ 
                                  loan_grade_test_WOE + 
                                  loan_int_rate_test_WOE +
                                  loan_percent_income_test_WOE+
                                  person_income_test_WOE +
                                 loan_grade_test_WOE*loan_int_rate_test_WOE+
                                 loan_grade_test_WOE*loan_percent_income_test_WOE
                                ,data = df_binned_test, family = "binomial")


# Step 6: Apply the logistic model to the test data
df_binned_test$predicted_prob_test <- predict(logit_model_binned_test, 
                                              newdata = df_binned_test, 
                                              type = "response")

# Ensure loan_status remains numeric (0 and 1) in the test set
df_binned_test$loan_status <- as.numeric(as.character(df_binned_test$loan_status))

# Sort df_binned_test by predicted_prob_test in descending order and group by predicted_prob_test
df_binned_test_grouped <- df_binned_test %>%
  arrange(desc(predicted_prob_test)) %>%
  group_by(predicted_prob_test) %>%
  summarise(count_test = n(),
            bad_count_test = sum(loan_status))

# Sort df_binned_test_grouped by predicted_prob_test in descending order
df_binned_test_grouped <- df_binned_test_grouped %>%
  arrange(desc(predicted_prob_test))

# Add initial point (0, 0) for CAP curve
initial_point_test <- data.frame(predicted_prob_test = 0, count_test = 0, bad_count_test = 0)
df_binned_test_grouped <- bind_rows(initial_point_test, df_binned_test_grouped)

# Calculate cumulative bad samples and total samples for test data
df_binned_test_grouped <- df_binned_test_grouped %>%
  mutate(cum_bad_test = cumsum(bad_count_test),
         cum_total_test = cumsum(count_test),
         total_bad_test = sum(df_binned_test$loan_status),
         total_count_test = sum(df_binned_test_grouped$count_test))

# Calculate CAP curve data for test data
df_binned_test_grouped <- df_binned_test_grouped %>%
  mutate(cum_perc_bad_test = round(cum_bad_test / total_bad_test, 3),
         cum_perc_total_test = round(cum_total_test / total_count_test, 3))

# Calculate perfect model data for test data
perfect_model_test <- df_binned_test %>%
  arrange(desc(loan_status)) %>%
  mutate(cum_bad_test = cumsum(loan_status),
         total_bad_test = sum(loan_status),
         cum_perc_bad_test = cum_bad_test / total_bad_test,
         cum_perc_total_test = row_number() / n())

# Step 7: Calculate AR for the logistic model on test data
ar_logit_test <- calculate_ar(df_binned_test_grouped, 
                              "cum_perc_total_test", 
                              "cum_perc_bad_test", 
                              df_binned_test_grouped$total_bad_test[1],
                              df_binned_test_grouped$total_count_test[1])

# Plot CAP curves for both training and test data and add AR annotations
ggplot() +
  # Training data CAP curve
  geom_line(data = df_binned_train_grouped, aes(x = cum_perc_total, y = cum_perc_bad), 
            color = "blue", linewidth = 1) +
  # Test data CAP curve
  geom_line(data = df_binned_test_grouped, aes(x = cum_perc_total_test, y = cum_perc_bad_test), 
            color = "orange", linewidth = 1) +
  # Perfect model curve (for training data)
  geom_line(data = perfect_model, aes(x = cum_perc_total, y = cum_perc_bad), 
            color = "green", linetype = "dashed", linewidth = 1) +
  # Perfect model curve (for test data)
  geom_line(data = perfect_model_test, aes(x = cum_perc_total_test, y = cum_perc_bad_test), 
            color = "green", linetype = "dashed", linewidth = 1) +
  # Random model line
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dotted", linewidth = 1) +
  # Labels and annotations
  labs(title = "CAP Curve for predicted_prob on Training and Test Data",
       x = "Cumulative Percentage of Sorted Samples",
       y = "Cumulative Percentage of Bad Captured") +
  annotate("text", x = 0.5, y = 0.28, 
           label = paste("Training AR:", round(ar_logit_train, 3)), 
           color = "blue", size = 5, hjust = 0) +
  annotate("text", x = 0.5, y = 0.2, 
           label = paste("Test AR:", round(ar_logit_test, 3)), 
           color = "orange", size = 5, hjust = 0) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
