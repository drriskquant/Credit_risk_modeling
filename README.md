Credit Risk Modeling Project

This project involves credit risk modeling using a dataset sourced from Kaggle. The data file is named "credit_risk_dataset.csv," and it contains 32,581 rows with 11 predictor variables and 1 response variable. The entire project is conducted using RStudio.

Dataset Source:

Kaggle - Credit Risk Dataset https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data

Project Outline

Project 1: Data Analysis

Conduct graphical analysis for each variable individually and for each pair of (Response, Predictor).
This analysis helps in understanding the data distribution and identifying potential relationships between predictors and the response.

Project 2: PD Segmentation

Data Preprocessing:
Handle missing values using K-Nearest Neighbors (KNN) imputation.

Data Splitting:
Use the validation set method to split the dataset into a training set (70%) and a test set (30%).

Logistic Regression:
Perform logistic regression on the training set and generate predictions (pred_logit).

Segmentation:
Use the prediction from the logistic regression model as a new driver and build a decision tree model using only this learner to segment the population. Each leaf node in the tree represents a homogeneous segment.

CAP Plot:
Draw the CAP (Cumulative Accuracy Profile) plot for the new model and compare it with the previous model.
[Uploading CAP_plot_comparison…]()

Segmentation Adjustment:
Adjust the segmentation based on the number of segments and the accuracy ratio to balance these factors.

Default Rate Analysis:
Display the default rate within each segment for both the training and test sets.

Comparison:
Compare the classification results between the training and test sets using CAP plots.

Project 3: Variable Selection

Use Random Forest to evaluate the importance of all predictor variables, using the Gini Index as the importance measure.

Select the top 5 most important variables for further analysis.

Project 4: Binning and WOE Transformation

Binning:
Partition each predictor variable using a decision tree based on the original response variable.

New Driver:
Use the predictions from the binning process as new drivers to represent the corresponding predictor variables.

WOE Transformation:
Perform Weight of Evidence (WOE) transformation (logit transformation) on each predictor.

Logistic Regression:
Build a logistic regression model using the original response and the newly defined drivers based on the WOE transformation.

Tools & Libraries

RStudio

Libraries: ggplot2, randomForest, caret, rpart, etc.
