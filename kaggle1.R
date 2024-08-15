# Load necessary libraries
library(ggplot2)
library(dplyr)

# Load data
df <- read.csv("credit_risk_dataset.csv")

num_vars <- sapply(df, is.numeric)
cat_vars <- sapply(df, is.factor) | sapply(df, is.character)

# Delete outliers
df <- df[df$person_age <= 100, ]
df <- df[df$person_emp_length <= 50, ]

print(df)

# Plot histogram for numerical variables
lapply(names(df)[num_vars], function(col) {
  # Calculate the histogram data
  hist_data <- hist(df[[col]], breaks = 10, plot = FALSE)
  # Convert frequencies to percentages
  hist_data$counts <- hist_data$counts / sum(hist_data$counts) * 100
  # Plot the histogram with percentages
  plot(hist_data, main = paste("Histogram of", col), xlab = col, col = "blue", ylab = "Percentage")
})

# Plot barplot for categorical variables
lapply(names(df)[cat_vars], function(col) {
  # Calculate the frequency table
  freq_table <- table(df[[col]])
  # Convert frequencies to percentages
  percent_table <- prop.table(freq_table) * 100
  # Plot the barplot with percentages
  barplot(percent_table, 
          main = paste("Barplot of", col), 
          xlab = col, 
          ylab = "Percentage", 
          col = "blue")