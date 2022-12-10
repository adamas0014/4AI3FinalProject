#!/usr/bin/env python3

from sklearn.svm import SVC

def Supportvector(xTrainSc,yTrain,xTestSc,yTest):
    svm = SVC()
    svm.fit(xTrainSc, yTrain)
    svmScore = svm.score(xTestSc,yTest)
    return(svmScore)