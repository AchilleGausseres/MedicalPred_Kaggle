import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv(
    "/Users/achillegausseres/OneDrive/PRO/Kaggle/Prediction_Medic/MedicalPred_Kaggle_RepoGit/medical_conditions_dataset.csv")
data.head()

data.info()

data.describe()

print(data.isnull().sum())

from missforest import MissForest

data = data.drop(["id", "full_name"], axis=1)
newdata = MissForest()
newdata.fit(
    x=data)
print(newdata.isnull().sum())


