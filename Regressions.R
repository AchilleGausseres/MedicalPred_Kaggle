setwd("~/OneDrive/PRO/Kaggle/Prediction_Medic/MedicalPred_Kaggle_RepoGit")
load("data_imputed.Rdata")
final_impute
set.seed(123)

## Split dataset ##
indice <- seq(1, length(final_impute[,1]))
train_indices <- sample(indice, size = length(final_impute[,1])*0.8)
data_train <- final_impute[train_indices,]
data_test <- final_impute[-train_indices,]
## ---------- ## 


library(nnet)
mod.glm <- multinom(condition ~ ., data=data_train)
summary(data_train)
glm.pred <- predict(mod.glm, newdata = data_test[,-7])

library(caret)
confusionMatrix(glm.pred, data_test[,7])

summary(glm.pred)
summary(data_train)
summary(data_test)

library(MASS)

step(mod.glm, direction = "forward")


mod_lda <- lda(condition ~., data = data_train)
lda.pred <- predict(mod_lda,data_test[,-7])$class
lda.pred
summary(lda.pred)
confusionMatrix(lda.pred, data_test[,7])

library(ranger)
mod.forest <- ranger(formula = condition~., 
                     data=data_train)
forest.pred <- predict(mod.forest, data_train[,-7], type = "response")

head(forest.pred)
confusionMatrix(forest.pred$predictions, data_train[,7])


acc_test <- c()
test_values <- c(2,100, 500, 1000, 5000, 10000,20000,30000)
for (t in test_values){
  mod.forest <- ranger(formula = condition~., 
                       num.trees = t,
                       data=data_train)
  forest.pred <- predict(mod.forest, data_train[,-7], type = "response")
  acc_test <- c(acc_test, confusionMatrix(forest.pred$predictions, data_train[,7])$overall[1])
  print(t)
}
plot(test_values, acc_test, type = 'l')


## Best model 
mod.forest <- ranger(formula = condition~., 
                     num.trees = 5000,
                     data=data_train, 
                     importance = "impurity")
forest.pred <- predict(mod.forest, data_train[,-7], type = "response")

head(forest.pred)
confusionMatrix(forest.pred$predictions, data_train[,7])
mod.forest$variable.importance


## Add weights to class
weights_ranger <- c(8000/1177, 8000/4799, 8000/2024)

mod.forest <- ranger(formula = condition~., 
                     num.trees = 5000,
                     data=data_train, 
                     importance = "impurity", 
                     class.weights = weights_ranger)
forest.pred <- predict(mod.forest, data_train[,-7], type = "response")

head(forest.pred)
confusionMatrix(forest.pred$predictions, data_train[,7])$overall[1]
confusionMatrix(forest.pred$predictions, data_train[,7])$table


