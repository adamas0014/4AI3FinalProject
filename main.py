#!/usr/bin/env python3
import numpy as np
import pandas as pd
import glob
import time
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt



datasetPath = "C:\\Users\\adama\\OneDrive\\Documents\\4AI3_FinalProject\\dataset\\"
normal = glob.glob(datasetPath + 'normal\\normal\\*.csv', recursive=True)
imbalance6g = glob.glob(datasetPath + 'imbalance\\imbalance\\6g\\*.csv')
imbalance10g = glob.glob(datasetPath + 'imbalance\\imbalance\\10g\\*.csv')
imbalance15g = glob.glob(datasetPath + 'imbalance\\imbalance\\15g\\*.csv')
imbalance20g = glob.glob(datasetPath + 'imbalance\\imbalance\\20g\\*.csv')
imbalance25g = glob.glob(datasetPath + 'imbalance\\imbalance\\25g\\*.csv')
imbalance30g = glob.glob(datasetPath + 'imbalance\\imbalance\\30g\\*.csv')
imbalance35g = glob.glob(datasetPath + 'imbalance\\imbalance\\35g\\*.csv')

def CSV2Dataset(pathList):
    df = pd.DataFrame()
    for x in pathList:
        inData = pd.read_csv(x, header=None)
        df = pd.concat([df, inData], ignore_index = True)
    
    return df




def visualizeDF2(df, category, colNames):
    figure, axList = plt.subplots(8, sharex=False, sharey=False,figsize=(20,20))
    figure.suptitle(category)
    figure.tight_layout(pad = 5)
    
    for index, i in enumerate(df.columns):
        axList[i].plot(df[i])
        axList[i].set_title(colNames[index])
    
    plt.savefig(f'./plots/{category}.png', dpi=400)




def routine():
    colNames = ["Tachometer Signal", "Underhang Accel. X",  "Underhang Accel. Y",  "Underhang Accel. Z",  "Overhang Accel. X",  "Overhang Accel. Y",  "Overhang Accel. Z", "Microphone"]
    plots = {"Normal - Baseline": normal, "Imbalance - 6g": imbalance6g, "Imbalance - 10g": imbalance10g, 
        "Imbalance - 15g": imbalance15g, "Imbalance - 20g": imbalance20g, "Imbalance - 25g": imbalance25g, 
        "Imbalance - 30g": imbalance30g, "Imbalance - 35g": imbalance35g }
    
    for key, value in plots.items():
        focusDf = CSV2Dataset(value)
        print(focusDf.head())
        visualizeDF2(focusDf, key, colNames)
        print(f'Processed {key}')
    


if __name__ == "__main__":
    routine()

