#!/usr/bin/env python3

from tensorflow import keras


def NeuralNetwork(xTrainSc,yTrain,xTestSc,yTest):
    units= 256
    epoch = 1
    optimizers = 'adam'
    model = keras.Sequential([keras.layers.Dense(units, activation='relu', input_shape=[xTrainSc.shape[1]]),
                              keras.layers.Dense(1, activation = 'sigmoid')])
    model.compile(optimizer=optimizers, loss= 'binary_crossentropy',
                    metrics=['binary_accuracy', 'AUC'])
    history = model.fit(xTrainSc, yTrain, epochs=epoch, verbose=1, validation_split=0.1)
    score = model.evaluate(xTestSc, yTest)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'])
    plt.show()
    return score