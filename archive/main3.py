#!/usr/bin/env python3
import numpy as np
import pandas as pd
import glob
import time
import seaborn as sns
import sys
import matplotlib
matplotlib.use('Agg')
import sklearn
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import pickle

datasetPath = "C:\\Users\\adama\\OneDrive\\Documents\\4AI3_FinalProject\\dataset\\"
normal = glob.glob(datasetPath + 'normal\\normal\\*.csv', recursive=True)
imbalance6g = glob.glob(datasetPath + 'imbalance\\imbalance\\6g\\*.csv')
imbalance10g = glob.glob(datasetPath + 'imbalance\\imbalance\\10g\\*.csv')
imbalance15g = glob.glob(datasetPath + 'imbalance\\imbalance\\15g\\*.csv')
imbalance20g = glob.glob(datasetPath + 'imbalance\\imbalance\\20g\\*.csv')
imbalance25g = glob.glob(datasetPath + 'imbalance\\imbalance\\25g\\*.csv')
imbalance30g = glob.glob(datasetPath + 'imbalance\\imbalance\\30g\\*.csv')
imbalance35g = glob.glob(datasetPath + 'imbalance\\imbalance\\35g\\*.csv')

def CSV2Dataset(file, colNames):
    print("retreiving dataset")
    df = pd.DataFrame()
    for x in file:
        inData = pd.read_csv(x, header=None, names = colNames)
        df = pd.concat([df, inData], ignore_index = True)
    
    return df

def downSample(df, samplingRate):
    print("downsampling...")
    startItr = 0
    endItr = samplingRate
    outDf = pd.DataFrame()

    for i in range(int(len(df) / samplingRate)):
        outDf = outDf.append(df.iloc[startItr: endItr].sum() / samplingRate, ignore_index = True)
        startItr += samplingRate
        endItr += samplingRate
        if startItr % 250000 == 0: print(startItr)
    return outDf


def buildDataframe():
    samplingRate = 200
    masterDf = pd.DataFrame()

    datasets = {"Normal": normal, "Imbalance6g": imbalance6g, "Imbalance10g": imbalance10g, 
        "Imbalance15g": imbalance15g, "Imbalance20g": imbalance20g, "Imbalance25g": imbalance25g, 
        "Imbalance30g": imbalance30g, "Imbalance35g": imbalance35g }
    columnNames = ["Tachometer Signal", "Underhang Accel. X",  "Underhang Accel. Y",  "Underhang Accel. Z",  "Overhang Accel. X",  "Overhang Accel. Y",  "Overhang Accel. Z", "Microphone"]
    itr = 0
    for key, value in datasets.items():
        print(f'starting on {key}')
        focusDf = CSV2Dataset(value, columnNames)
        focusDf = downSample(focusDf, samplingRate)
        if itr == 0:
            focusDf['Class'] = 0
        else:
            focusDf['Class'] = 1
    
        print(f'Finished processing {key}')
        focusDf.to_csv(f'sampledDatasets/{samplingRate}/{key}.csv', index=False)
        focusDf.head()
        masterDf = masterDf.append(focusDf, ignore_index=True)
        del focusDf
        print(f'Appended {key}')
        itr +=1

    masterDf.head()
    masterDf.to_csv(f'sampledDatasets/{samplingRate}/master{samplingRate}.csv', index=False)


def oneHotEncodeClass():
    df = pd.read_csv('sampledDatasets/200/master200.csv')
    dfOHE = pd.get_dummies(df['Class'])
    df = df.drop('Class', axis = 1)
    df = df.join(dfOHE)
    df.to_csv(f'sampledDatasets/200/master200OHE.csv', index=False)

def prepareAndScaleData(df):
    #df = df.sample(frac = 1, random_state = 7).reset_index()
    print(df.head())
    y = df['Class']
    X = df.drop('Class', axis = 1)
    print(X.head())
    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, y, shuffle = True, test_size = 0.2, random_state = 10)
    xTrain.head()
    xTrain = xTrain.drop(['Tachometer Signal','Underhang Accel. Y', 'Underhang Accel. Z'], axis = 1)
    xTest = xTest.drop(['Tachometer Signal','Underhang Accel. Y', 'Underhang Accel. Z'], axis = 1)
    print(f'xTrain: {xTrain.shape}\nyTrain: {yTrain.shape}\nxTest: {xTest.shape}\nyTest: {yTest.shape}\n')

    sc = sklearn.preprocessing.StandardScaler()
    xTrainSc = sc.fit_transform(xTrain)
    xTestSc = sc.fit_transform(xTest)
    print(xTrainSc[:5])
    return xTrainSc, yTrain, xTestSc, yTest


hyperparameterProfiles = dict({

    "A": {
        "units": 128,
        "epochs": 10,
        "optimizers": 'adam',
        "regularizer": 'regularizers.l2(0.01)'
    },
        
})

def hyperParameterTuning(xTrain, yTrain, hpProfiles, xTest, yTest):
    
    for key, val in hpProfiles.items():
        
        X_reg_new=SelectKBest(score_func=f_regression, k=5)
        X_train_selected = X_reg_new.fit_transform(xTrain,yTrain)
        print(X_train_selected[:5])

        print(f'Starting profile {key}: [ u: { val["units"] }, e: { val["epochs"] }, o: { val["optimizers"] }, r: { val["regularizer"] } ]')
        model = keras.Sequential([
            keras.layers.Dense(val['units'], kernel_regularizer = regularizers.L2(0.01), activation = 'relu', input_shape = [xTrain.shape[1]]),
            keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(
            optimizer = val['optimizers'],
            loss = 'binary_crossentropy',
            metrics = 'accuracy'
        )
        history = model.fit(xTrain, yTrain, verbose = 1, epochs = val['epochs'], validation_split = 0.1)

        model.evaluate(xTest, yTest)
        myModelYPredict = [0 if x < 0.5 else 1 for x in model.predict(xTest)]
        myModelAccuracy = sklearn.metrics.accuracy_score(yTest, myModelYPredict)
        print(f'Profile {key}: [ u: { val["units"] }, e: { val["epochs"] }, o: { val["optimizers"] } ] -> Accuracy: {myModelAccuracy}', file = open('outData.txt', 'a'))

def processNeuralModel():
    inData = pd.read_csv('sampledDatasets/200/master200.csv')
    xTrain, yTrain, xTest, yTest = prepareAndScaleData(inData)
    hyperParameterTuning(xTrain, yTrain, hyperparameterProfiles, xTest, yTest)


if __name__ == "__main__":
    processNeuralModel()
