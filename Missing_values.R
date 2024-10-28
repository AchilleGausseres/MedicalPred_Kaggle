## Libraries ##
library(ggplot2)
library(VIM)
library(missRanger)
library(missMDA)
## --------- ##

## Import dataset ##
setwd("~/OneDrive/PRO/Kaggle/Prediction_Medic/MedicalPred_Kaggle_RepoGit")
data <- read.csv(file="medical_conditions_dataset.csv", 
          header = T, 
          sep = ",", 
          dec = ".",
          stringsAsFactors = T)
summary(data)
## --------- ##

## Missing values visualization ##
countNA(data)
aggr(data)
matrixplot(data, sortby = "condition", interactive = F)
## --------- ##

## Compare imputation methods ##
set.seed(123)
data <- data[, -(1:2)]

data_origin <- data_imp <- data
for (i in 1:ncol(data_imp)){
  data_imp[,i][sample(1:nrow(data), 500)] <- NA
}
missing_values <- is.na(data_imp)
data_ForestImp <- missRanger(data_imp)
data_KnnImp <- kNN(data_imp)
data_IrmiImp <- irmi(data_imp)
type <- c()
for (i in 1:ncol(data_origin)){
  type <- c(type, class(data_origin[23,i]))
}

evaluation(data_origin, data_ForestImp, m = missing_values, 
           vartypes = type)
evaluation(data_origin, data_KnnImp[,-(8:14)], m = missing_values, 
           vartypes = type)
evaluation(data_origin, data_IrmiImp[,-(8:14)], m = missing_values, 
           vartypes = type)
## --------- ##

## Final imputation ##
final_impute <- missRanger(data, 
                           num.trees = 1000, seed=1, pmm.k = 10)
View(final_impute)
## --------- ##

save(final_impute, file ="data_imputed.Rdata") 

