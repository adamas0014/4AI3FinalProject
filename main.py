# %%Data import and outlier removal
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import glob
import time
import seaborn as sns
import sys
import matplotlib
import sklearn
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from NeuralNet import *
from KNeighbors import *
from SVCs import SupportVectors


#import data
df = pd.read_csv('master200.csv')
rand = 12
df = shuffle(df, random_state = rand).reset_index().drop('index', axis = 1)
#print(df.head())

#standard scaler on the data
y = df['Class']
X = df.drop(['Class'], axis = 1)
colnames = X.columns
standard = StandardScaler()
X = standard.fit_transform(X)
X=pd.DataFrame(X,columns = colnames)

#Removes all values less than 3
X['class'] = y
colnames = X.columns
for i in range(len(colnames)):
  X = X.loc[X[colnames[i]]<=3]

#Upsampled data to balance dataset
print(f'Count of null values in dataset: {df.isnull().sum().sum()}')
y = df['Class']
X = df.drop(['Class'], axis = 1)
print(f'BEFORE: Count by class {y.value_counts()}\n')
from imblearn.over_sampling import RandomOverSampler
overSampler = RandomOverSampler(sampling_strategy = 'not majority')
X_adj, y_adj = overSampler.fit_resample(X, y)
print(f'AFTER: Count by class {y_adj.value_counts()}')

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X_adj, y_adj, shuffle = True, test_size = 0.2, random_state = 10)
print(f'xTrain: {xTrain.shape}\nyTrain: {yTrain.shape}\nxTest: {xTest.shape}\nyTest: {yTest.shape}\n')

sc = sklearn.preprocessing.StandardScaler()
xTrainSc = sc.fit_transform(xTrain)
xTestSc = sc.fit_transform(xTest)
print(f'Scaled: [ XTrain: {xTrainSc.shape}, XTest: {xTestSc.shape} ] \n')

# %% Kneighbors Classifier
kScore = KNeighbors(xTrainSc,yTrain,xTestSc,yTest)
print(kScore)
# %% Neural Net
netScore = NeuralNetwork(xTrainSc,yTrain,xTestSc,yTest)
print(netScore)
# %% SVC classifier (don't run--> takes >5 hours)
svcScore = SupportVectors(xTrainSc,yTrain,xTestSc,yTest)
print(svcScore)

# %%
