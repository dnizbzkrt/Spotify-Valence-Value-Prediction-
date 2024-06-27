
library(tidyverse)
library(ggplot2)
library(naniar)
library(corrplot)
library(ggcorrplot)
library(caret)
library(VIM)
library(dplyr)


df <- read.csv("filename.csv")
set.seed(412)
new_df <- df[sample(nrow(df), size=1000), ]
library(missForest)
df = prodNA(new_df, noNA = 0.05)
colnames(df)
df <- df[,!names(df) %in% c("track_name", "lyrics","mode")]
colnames(df)[1:2] <- c("artist","genre")
sum(duplicated(df))

#cleaned

#derscriptive
summary(df)
max(table(df$artist)) # most occured artist frequency is 6
length(unique(df$artist))
table(df$genre)
ggplot(df, aes(x = genre)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Genre Count", x = "Genre", y = "Count")


# Correlation matrix
numeric_cols <- df %>% select(where(is.numeric))
cor_matrix <- cor(numeric_cols, use = "pairwise.complete.obs")

# Plot correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", number.cex = 0.5)
#energy and loudness
library(GGally)
ggpairs(numeric_cols)
###
library(mice)
# Visualize missing data pattern
vis_miss(df)
colSums(is.na(df))

imputed_Data <- mice(df, m=1, maxit = 1, method ='pmm', seed = 412)
complete_data<-mice::complete(imputed_Data,1)
colSums(is.na(complete_data))
fill_mode <- function(column) {
  mode_value <- names(sort(table(column), decreasing = TRUE))[1]
  column[is.na(column)] <- mode_value
  return(column)
}



complete_data <- complete_data %>%
  mutate(
    artist = fill_mode(artist),
    genre = fill_mode(genre)
  )

###
create_density_plot <- function(data, variable_name, title) {
  ggplot(data, aes_string(x = variable_name)) +
    geom_density(na.rm = TRUE) +
    ggtitle(title) +
    theme_minimal()
}
plot_list <- lapply(names(df)[3:12], function(var) {
  create_density_plot(df, var, paste("Kernel Density Before"))
})
plot_list_imputed <- lapply(names(complete_data)[3:12], function(var) {
  create_density_plot(complete_data, var, paste("Kernel Density After"))
})
for (i in 1:length(plot_list)) {
  grid.arrange(plot_list[[i]], plot_list_imputed[[i]], ncol = 2)
}

###
#violin
# Filter out rows with non-missing valence values
df_filtered <- df[complete.cases(df[, c("genre", "valence")]), ]

# Violin plot of valence by genre (excluding NA values)
ggplot(df_filtered, aes(x = genre, y = valence, fill = genre)) +
  geom_violin(trim = FALSE) +
  labs(title = "Valence by Genre", x = "Genre", y = "Valence") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#seems no sig difference in valence mean between  genre 


boxplot_list <- lapply(names(numeric_cols), function(var) {
  ggplot(df, aes_string(y = var)) +
    geom_boxplot() +
    labs(title = paste("Box Plot of", var),
         y = var) +
    theme_minimal()
})
#valence 

# Arrange box plots in a grid
library(gridExtra)
grid.arrange(grobs = boxplot_list, ncol = 3)
#seems there are outliers

########
library(MASS)
str(complete_data)
numeric_cols <- complete_data[,c(3:12)]
head(numeric_cols)
detect_outliers <- function(x) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = TRUE)
  H <- 1.5 * IQR(x)
  outliers <- x < (qnt[1] - H) | x > (qnt[2] + H)
  outliers
}
nrow(complete_data[apply(apply(numeric_cols, 1, detect_outliers), 1, any), ])
#so many outliers we assume no outliers


library(MASS)
qqnorm(complete_data$energy)
shapiro.test(complete_data$valence)
summary(rlm(energy ~ loudness, data = complete_data))
#tvalue 29.25 for loudness  and 86.93  for intercept
#there is multicollenarity

#violin
shapiro.test(complete_data$valence)#not normal
#valence p shapiro is p<0.05

kruskal.test(valence ~ genre, data = complete_data)
library(dunn.test)
dunn.test(complete_data$valence, complete_data$genre, method = "bonferroni")
#significant difference in valence mean in genre
#EDM different others but others are not  different each  other.
#for  edm- pvalues (<0.05)


ggpairs(complete_data%>% select(where(is.numeric)))

#define your one hot encoding function 
dummy <- dummyVars(" ~ .", data=complete_data)
head(dummy)
dim(complete_data)
final_df <- data.frame(predict(dummy, newdata=complete_data))
dim(final_df)


library(Boruta)
set.seed(412)
boruta.cereal_train <- Boruta(valence~., data = final_df, doTrace = 2)
print(boruta.cereal_train)
boruta.cereal <- TentativeRoughFix(boruta.cereal_train)
print(boruta.cereal)
plot(boruta.cereal, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.cereal$ImpHistory),function(i)
  boruta.cereal$ImpHistory[is.finite(boruta.cereal$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.cereal$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.cereal$ImpHistory), cex.axis = 0.7)
getSelectedAttributes(boruta.cereal, withTentative = F)
df_selected <- final_df %>% dplyr::select(all_of(c(getSelectedAttributes(boruta.cereal, withTentative = F),"valence")))


colSums(is.na(df_selected))
str(df_selected)
#check linearity
library(car)








# Split the data into training and test set
set.seed(412)
training.samples <- df_selected$valence %>% createDataPartition(p = 0.8, list = FALSE) #createDataPartition helps you define train set index 
train.data  <- df_selected[training.samples, ]
test.data <- df_selected[-training.samples, ]
fitcontrol<- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 5)
#glm
shapiro.test(sqrt(train.data$valence))
str(train.data)
glm_model <- glm(valence ~ ., data = train.data, family = gaussian)

p_values <- summary(glm_model)$coefficients[, "Pr(>|t|)"]

# Identify predictors with p-values above the significance level
predictors_to_remove <- names(p_values[p_values > 0.05])

# Remove unnecessary predictors from the data frame
train.data_filtered <- train.data[, !names(train.data) %in% predictors_to_remove]

# Fit the GLM model with filtered predictors
glm_model_filtered <- glm(valence ~ ., data = train.data_filtered, family = gaussian)

# Summary of the filtered model
summary(glm_model_filtered)
glm_model_cv <- train(valence ~ .,                   # Formula for the model
                      data = df_selected,           # Data
                      method = "glm",               # Method: Generalized Linear Model
                      trControl = fitcontrol,       # Train control parameters
                      preProcess = c("center", "scale"))  # Preprocessing: centering and scaling
train_predictions <- predict(glm_model, newdata = train.data)
test_predictions <- predict(glm_model, newdata = test.data)

# Compute RMSE and MAE for training and test sets
train_rmse <- RMSE(train_predictions, train.data$valence)
test_rmse <- RMSE(test_predictions, test.data$valence)

train_mae <- MAE(train_predictions, train.data$valence)
test_mae <- MAE(test_predictions, test.data$valence)

# Print the results
cat("Training RMSE:", train_rmse, "\n")
cat("Test RMSE:", test_rmse, "\n")

cat("Training MAE:", train_mae, "\n")
cat("Test MAE:", test_mae, "\n")

#Predicting
numx <- nrow(test.data)
x_axis <- seq(numx)
test_predictions<-predict(glm_model,newdata = test.data)
df <- data.frame(x_axis, test_predictions,test.data$valence)
#Plotting the predicted values against the actual values
g <- ggplot(df, aes(x = x_axis))
g <- g + geom_line(aes(y = test_predictions, colour = "Predicted")) +
  geom_point(aes(y = test_predictions, colour = "Predicted")) +
  geom_line(aes(y = test.data$valence, colour = "Actual")) +
  geom_point(aes(y = test.data$valence, colour = "Actual")) +
  scale_colour_manual("", values = c(Predicted = "red", Actual = "blue")) +
  labs(x = "Sample's Number", y = "Valence") +
  ggtitle("Valence Predicted vs Actual Values")

g



#ANN
set.seed(412)
nn_model <-train(valence ~.,  data =train.data, method = "nnet", trControl = fitcontrol)
nn_model$bestTune
summary(nn_model)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
plot.nnet(nn_model$finalModel)
train_predictions <- predict(nn_model, newdata = train.data)
test_predictions <- predict(nn_model, newdata = test.data)

# Compute RMSE and MAE for training and test sets
train_rmse <- sqrt(mean((train_predictions - train.data$valence)^2))
test_rmse <- sqrt(mean((test_predictions - test.data$valence)^2))

train_mae <- mean(abs(train_predictions - train.data$valence))
test_mae <- mean(abs(test_predictions - test.data$valence))

# Print the results
cat("Training RMSE:", train_rmse, "\n")
cat("Test RMSE:", test_rmse, "\n")

cat("Training MAE:", train_mae, "\n")
cat("Test MAE:", test_mae, "\n")
#svm
para_value_min <- c( gamma = 1e-3, c = 1e-4 )
para_value_max <- c( gamma = 2, c = 10 )
grid_svmlinear  <- expand.grid(C = seq(.25,1,.25) )
svmlinearFit<- train(valence~.,data = train.data,
                     trControl=fitcontrol ,
                     method = "svmLinear",
                     preProc = c("center", "scale"), savePredictions = T, tuneGrid = grid_svmlinear)
grid_svmradial <- expand.grid(C = seq(0.25, 1, by = 0.25), 
                              sigma = seq(0.01, 0.1, by = 0.01))
svmradialFit <- train(valence ~ ., data = train.data,
                      trControl = fitcontrol,
                      method = "svmRadial",
                      preProc = c("center", "scale"),
                      savePredictions = TRUE,
                      tuneGrid = grid_svmradial)
svmradialFit$bestTune
plot(svmradialFit)
# Compute RMSE and MAE for training and test sets
train_predictions <- predict(svmradialFit, newdata = train.data)
test_predictions <- predict(svmradialFit, newdata = test.data)

train_rmse <- sqrt(mean((train_predictions - train.data$valence)^2))
train_mae <- mean(abs(train_predictions - train.data$valence))

test_rmse <- sqrt(mean((test_predictions - test.data$valence)^2))
test_mae <- mean(abs(test_predictions - test.data$valence))

# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train MAE:", train_mae, "\n")

cat("Test RMSE:", test_rmse, "\n")
cat("Test MAE:", test_mae, "\n")

#random

#randomforest
set.seed(412)
library(randomForest)
rfFit <- train(valence~.,data = train.data,  method = "rf",trControl = fitcontrol,metric="MAE",
               )
set.seed(412)
tune_mtry <- expand.grid(.mtry = c(1: 7)) #Creates a vector with name ".mtry" (must have the same name as the parameter you're trying to tune)
mtry <- train(valence~.,data = train.data,
              method = "rf",
              metric = "MAE",
              tuneGrid = tune_mtry,
              trControl = fitcontrol,
              importance = TRUE,
              nodesize = 14,
              ntree = 10)
print(mtry)
best_mtry = mtry$bestTune$mtry
best_mtry
mn_list <- list()
tune_mtry <- expand.grid(.mtry = best_mtry) #Using only the best value of mtry
for (mn in c(5:15)) {
  mNode <- train(valence~.,data = train.data,
                 method = "rf",
                 metric = "MAE",
                 tuneGrid = tune_mtry,
                 trControl = fitcontrol,
                 importance = TRUE,
                 nodesize = 14,
                 maxnodes = mn,
                 ntree = 10)
  i <- toString(mn)
  mn_list[[i]] <- mNode
}
result <- resamples(mn_list)
summary(result)
#using best mtry and maxnode
nt_list <- list()
for (nt in 1:10*100) {
  nTree <- train(valence~.,data = train.data,
                 method = "rf",
                 metric = "MAE",
                 tuneGrid = tune_mtry,
                 trControl = fitcontrol,
                 importance = TRUE,
                 nodesize = 14,
                 maxnodes = 12,
                 ntree = nt)
  i <- toString(nt)
  nt_list[[i]] <- nTree
}
result <- resamples(nt_list)
summary(result)
set.seed(412)
rfFit <- train(valence~.,data = train.data,  method = "rf",ntree=400,
               maxnodes=5,trControl = fitcontrol)
plot(rfFit)
# Predict on the training and test data
train_predictions <- predict(rfFit, newdata = train.data)
test_predictions <- predict(rfFit, newdata = test.data)

# Compute RMSE and MAE for training and test sets
train_rmse <- RMSE(train_predictions, train.data$valence)
train_mae <- MAE(train_predictions, train.data$valence)

test_rmse <- RMSE(test_predictions, test.data$valence)
test_mae <- MAE(test_predictions, test.data$valence)

# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train MAE:", train_mae, "\n")

cat("Test RMSE:", test_rmse, "\n")
cat("Test MAE:", test_mae, "\n")

#Predicting
numx <- nrow(train.data)

x_axis <- seq(numx)
dt_train_pred<-predict(rfFit,newdata = train.data)
df <- data.frame(x_axis, dt_train_pred,train.data$valence)
#Plotting the predicted values against the actual values
g <- ggplot(df, aes(x=x_axis))
g <- g + geom_line(aes(y=dt_train_pred, colour="Predicted"))
g <- g + geom_point(aes(x=x_axis, y=dt_train_pred, colour="Predicted"))
g <- g + geom_line(aes(y=train.data$valence, colour="Actual"))
g <- g + geom_point(aes(x=x_axis, y=train.data$valence, colour="Actual"))
g <- g + scale_colour_manual("", values = c(Predicted="red", Actual="blue"))
g
rf_var_imp<-varImp(rfFit, scale = FALSE)
rf_var_imp
plot(rf_var_imp)
test_prediction<-predict(rfFit,newdata= test.data)
numx <- nrow(test.data)
x_axis <- seq(numx)
df <- data.frame(x_axis,test_prediction,test.data$valence)
#Plotting the predicted values against the actual values
g <- ggplot(df, aes(x=x_axis))
g <- g + geom_line(aes(y=test_prediction, colour="Predicted"))
g <- g + geom_point(aes(x=x_axis, y=test_prediction, colour="Predicted"))
g <- g + geom_line(aes(y=test.data$valence, colour="Actual"))
g <- g + geom_point(aes(x=x_axis, y=test.data$valence, colour="Actual"))
g <- g + scale_colour_manual("", values = c(Predicted="red", Actual="blue"))
g
paste("MAE of Model:", MAE(as.numeric(test_prediction),test.data$valence))
train_MAE <- MAE(as.numeric(dt_train_pred), train.data$valence)
test_MAE <- MAE(as.numeric(test_prediction), test.data$valence)

# Create a bar plot
barplot(c(train_MAE, test_MAE), 
        names.arg = c("Training", "Testing"),
        col = c("blue", "red"),
        main = "Mean Absolute Error (MAE) Comparison",
        ylab = "MAE",
        ylim = c(0, max(train_MAE, test_MAE) * 1.2))

# Add text labels
text(1, train_MAE, round(train_MAE, 5), pos = 3, cex = 1, col = "blue")
text(2, test_MAE, round(test_MAE, 5), pos = 3, cex = 1, col = "red")
#gradient boosting
set.seed(412)
library(gbm)
gbmFit <- train(valence~.,data = train.data,  method = "gbm",trControl = fitcontrol,metric="MAE")
plot(gbmFit)
gbmFit$bestTune

#Predicting
numx <- nrow(train.data)
x_axis <- seq(numx)
gbm_train_pred<-predict(gbmFit,newdata = train.data)
df <- data.frame(x_axis, gbm_train_pred,train.data$valence)
#Plotting the predicted values against the actual values
g <- ggplot(df, aes(x=x_axis))
g <- g + geom_line(aes(y=gbm_train_pred, colour="Predicted"))
g <- g + geom_point(aes(x=x_axis, y=gbm_train_pred, colour="Predicted"))
g <- g + geom_line(aes(y=train.data$valence, colour="Actual"))
g <- g + geom_point(aes(x=x_axis, y=train.data$valence, colour="Actual"))
g <- g + scale_colour_manual("", values = c(Predicted="red", Actual="blue"))
g
paste("RMSE of Model:", RMSE(as.numeric(gbm_train_pred),train.data$valence))
gbm_var_imp<-varImp(gbmFit, scale = FALSE)
plot(gbm_var_imp)

test_prediction<-predict(gbmFit,newdata= test.data)
numx <- nrow(test.data)
x_axis <- seq(numx)
df <- data.frame(x_axis,test_prediction,test.data$valence)
#Plotting the predicted values against the actual values
g <- ggplot(df, aes(x=x_axis))
g <- g + geom_line(aes(y=test_prediction, colour="Predicted"))
g <- g + geom_point(aes(x=x_axis, y=test_prediction, colour="Predicted"))
g <- g + geom_line(aes(y=test.data$valence, colour="Actual"))
g <- g + geom_point(aes(x=x_axis, y=test.data$valence, colour="Actual"))
g <- g + scale_colour_manual("", values = c(Predicted="red", Actual="blue"))
g
paste("RMSE of Model:", RMSE(as.numeric(test_prediction),test.data$valence))
train_MAE <- MAE(as.numeric(gbm_train_pred), train.data$valence)
test_MAE <- MAE(as.numeric(test_prediction), test.data$valence)

# Create a bar plot
barplot(c(train_MAE, test_MAE), 
        names.arg = c("Training", "Testing"),
        col = c("blue", "red"),
        main = "Mean Absolute Error (MAE) Comparison",
        ylab = "MAE",
        ylim = c(0, max(train_MAE, test_MAE) * 1.2))

# Add text labels
text(1, train_MAE, round(train_MAE, 5), pos = 3, cex = 1, col = "blue")
text(2, test_MAE, round(test_MAE, 5), pos = 3, cex = 1, col = "red")


#xgboost
tune.gridxgb <- expand.grid(eta = c(0.05,0.3, 0.075), # 3 
                            nrounds = c(50, 75, 100),  # 3
                            max_depth = 4:7,  # 4
                            min_child_weight = c(2.0, 2.25), #2 
                            colsample_bytree = c(0.3, 0.4, 0.5), # 3
                            gamma = 0, #1
                            subsample = 1)  # 1
library(xgboost)
set.seed(412)
xgbFit <- train(valence~.,data = train.data,tuneGrid=tune.gridxgb,  method = "xgbTree",trControl = fitcontrol,metric="MAE")
xgbFit$bestTune
plot(xgbFit)
xgbFit <- train(valence ~ .,
                data = train.data,
                method = "xgbTree",
                trControl = fitcontrol,
                metric = "MAE",
                tuneGrid = data.frame(nrounds = 100,
                                      max_depth = 1,
                                      eta = 0.075,
                                      gamma = 0,
                                      colsample_bytree = 0.5,
                                      min_child_weight = 2,
                                      subsample = 1))
MAE(as.numeric(predict(xgbFit, newdata = train.data)), train.data$valence)
MAE(as.numeric(predict(xgbFit, newdata = test.data)
), test.data$valence)
xgb.plot.tree(model = xgbFit$finalModel, trees = 1)
library(DiagrammeR)


numx <- nrow(train.data)
x_axis <- seq(numx)
xgb_train_pred<-predict(xgbFit,newdata = train.data)
df <- data.frame(x_axis, xgb_train_pred,train.data$valence)
#Plotting the predicted values against the actual values
g <- ggplot(df, aes(x=x_axis))
g <- g + geom_line(aes(y=xgb_train_pred, colour="Predicted"))
g <- g + geom_point(aes(x=x_axis, y=xgb_train_pred, colour="Predicted"))
g <- g + geom_line(aes(y=train.data$valence, colour="Actual"))
g <- g + geom_point(aes(x=x_axis, y=train.data$valence, colour="Actual"))
g <- g + scale_colour_manual("", values = c(Predicted="red", Actual="blue"))
g
paste("RMSE of Model:", RMSE(as.numeric(xgb_train_pred),train.data$valence))
xgb_var_imp<-varImp(xgbFit, scale = FALSE)
plot(xgb_var_imp)
test_prediction<-predict(xgbFit,newdata= test.data)
numx <- nrow(test.data)
x_axis <- seq(numx)
df <- data.frame(x_axis,test_prediction,test.data$valence)
#Plotting the predicted values against the actual values
g <- ggplot(df, aes(x=x_axis))
g <- g + geom_line(aes(y=test_prediction, colour="Predicted"))
g <- g + geom_point(aes(x=x_axis, y=test_prediction, colour="Predicted"))
g <- g + geom_line(aes(y=test.data$valence, colour="Actual"))
g <- g + geom_point(aes(x=x_axis, y=test.data$valence, colour="Actual"))
g <- g + scale_colour_manual("", values = c(Predicted="red", Actual="blue"))
g
paste("RMSE of Model:", RMSE(as.numeric(test_prediction),test.data$valence))
train_MAE <- MAE(as.numeric(xgb_train), train.data$valence)
test_MAE <- MAE(as.numeric(test_prediction), test.data$valence)

# Create a bar plot
barplot(c(train_MAE, test_MAE), 
        names.arg = c("Training", "Testing"),
        col = c("blue", "red"),
        main = "Mean Absolute Error (MAE) Comparison",
        ylab = "MAE",
        ylim = c(0, max(train_MAE, test_MAE) * 1.2))

# Add text labels
text(1, train_MAE, round(train_MAE, 5), pos = 3, cex = 1, col = "blue")
text(2, test_MAE, round(test_MAE, 5), pos = 3, cex = 1, col = "red")






# Create separate prediction data frames for training and testing data
train_predictions <- data.frame(
  Actual = train.data$valence,
  glm_train = predict(glm_model, newdata = train.data),
  nn_train = predict(nn_model, newdata = train.data),
  svmlinear_train = predict(svmradialFit, newdata = train.data),
  rf_train = predict(rfFit, newdata = train.data),
  xgb_train = predict(xgbFit, newdata = train.data)
)

test_predictions <- data.frame(
  Actual = test.data$valence,
  glm_test = predict(glm_model, newdata = test.data),
  nn_test = predict(nn_model, newdata = test.data),
  svmlinear_test = predict(svmradialFit, newdata = test.data),
  rf_test = predict(rfFit, newdata = test.data),
  xgb_test = predict(xgbFit, newdata = test.data)
)
test_predictions
library(Metrics) # Load the Metrics package for performance metrics
library(caret) # Load caret for RB2 calculation

# Define a function to calculate regression metrics
calculate_regression_metrics <- function(actual, predicted) {
  rmse_value <- rmse(actual, predicted)
  mae_value <- mae(actual, predicted)
  r2_value <- R2(predicted, actual)
  return(c(RMSE = rmse_value, MAE = mae_value, R2 = r2_value))
}

# Compute metrics for each model on the training set
glm_train_metrics <- calculate_regression_metrics(train_predictions$Actual, train_predictions$glm_train)
nn_train_metrics <- calculate_regression_metrics(train_predictions$Actual, train_predictions$nn_train)
svmlinear_train_metrics <- calculate_regression_metrics(train_predictions$Actual, train_predictions$svmlinear_train)
rf_train_metrics <- calculate_regression_metrics(train_predictions$Actual, train_predictions$rf_train)
xgb_train_metrics <- calculate_regression_metrics(train_predictions$Actual, train_predictions$xgb_train)

# Compute metrics for each model on the testing set
glm_test_metrics <- calculate_regression_metrics(test_predictions$Actual, test_predictions$glm_test)
nn_test_metrics <- calculate_regression_metrics(test_predictions$Actual, test_predictions$nn_test)
svmlinear_test_metrics <- calculate_regression_metrics(test_predictions$Actual, test_predictions$svmlinear_test)
rf_test_metrics <- calculate_regression_metrics(test_predictions$Actual, test_predictions$rf_test)
xgb_test_metrics <- calculate_regression_metrics(test_predictions$Actual, test_predictions$xgb_test)

# Create data frames for training and testing metrics
train_metrics_df <- data.frame(
  Model = c("GLM", "NN", "SVM Linear", "Random Forest", "XGBoost"),
  RMSE = c(glm_train_metrics["RMSE"], nn_train_metrics["RMSE"], svmlinear_train_metrics["RMSE"], rf_train_metrics["RMSE"], xgb_train_metrics["RMSE"]),
  MAE = c(glm_train_metrics["MAE"], nn_train_metrics["MAE"], svmlinear_train_metrics["MAE"], rf_train_metrics["MAE"], xgb_train_metrics["MAE"]),
  R2 = c(glm_train_metrics["R2"], nn_train_metrics["R2"], svmlinear_train_metrics["R2"], rf_train_metrics["R2"], xgb_train_metrics["R2"])
)

test_metrics_df <- data.frame(
  Model = c("GLM", "NN", "SVM Linear", "Random Forest", "XGBoost"),
  RMSE = c(glm_test_metrics["RMSE"], nn_test_metrics["RMSE"], svmlinear_test_metrics["RMSE"], rf_test_metrics["RMSE"], xgb_test_metrics["RMSE"]),
  MAE = c(glm_test_metrics["MAE"], nn_test_metrics["MAE"], svmlinear_test_metrics["MAE"], rf_test_metrics["MAE"], xgb_test_metrics["MAE"]),
  R2 = c(glm_test_metrics["R2"], nn_test_metrics["R2"], svmlinear_test_metrics["R2"], rf_test_metrics["R2"], xgb_test_metrics["R2"])
)

# Print the results for training set
cat("Training Set Metrics:\n")
print(train_metrics_df)

# Print the results for testing set
cat("\nTesting Set Metrics:\n")
print(test_metrics_df)


glm_coefficients <- summary(glm_model)$coefficients
glm_importance <- data.frame(
  Variable = rownames(glm_coefficients),
  Coefficient = glm_coefficients[, "Estimate"],
  StdError = glm_coefficients[, "Std. Error"],
  tValue = glm_coefficients[, "t value"],
  pValue = glm_coefficients[, "Pr(>|t|)"]
)
glm_importance[order(abs(glm_importance$Coefficient), decreasing = TRUE), ]
ggplot(glm_importance, aes(x = reorder(Variable, abs(Coefficient)), y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip the coordinates for a horizontal bar plot
  xlab("Variable") +
  ylab("Coefficient") +
  ggtitle("Variable Importance for GLM Model") +
  theme_minimal()

importance <- varImp(glm_model, scale = FALSE)

# Create a bar plot
ggplot(importance, aes(x = reorder(rownames(importance), Overall), y = Overall)) +
  geom_bar(stat = "identity") +
  labs(x = "Predictor Variables", y = "Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
