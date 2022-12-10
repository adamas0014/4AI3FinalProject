from sklearn.neighbors import KNeighborsClassifier

def KNeighbors(xTrainSc,yTrain,xTestSc,yTest):
    leafsize = 30
    neighbors = 2
    pt=2
    #Create new KNN object
    knn = KNeighborsClassifier(leaf_size = leafsize, n_neighbors=neighbors, p=pt)
    #Fit the model
    best_model = knn.fit(xTrainSc, yTrain)
    knnScore = knn.score(xTestSc, yTest)
    return(knnScore)