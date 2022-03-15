from dataProcessing import getGoodData
from dataNormalization import DataModelation
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from tensorflow.keras.utils import to_categorical
import json

class NeuralNetworkModel:
    def __init__(self):
        with open('config.json','r') as file:
            configs = json.load(file)
            self.max_features = configs['max_features']
            n = configs['n']
            self.batch_size = configs['batch_size']
            normalization = configs['normalization']
            self.epochs = configs['epochs']

        data = None
        data = getGoodData(n)
        

        self.dataModelation = DataModelation(data, normalization, self.max_features)
        self.generateModel()
        self.trainModel()
       

    
    def generateModel(self): 
        self.model = Sequential()
        self.model.add(Input(shape=(self.dataModelation.matrix.shape[1]), batch_size=self.batch_size))
        self.model.add(Dense(64, activation="relu"))
        self.model.add( Dense(16, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(2, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def trainModel(self):
        self.model.fit( 
            self.dataModelation.matrix,
            to_categorical(self.dataModelation.labels,2),
            epochs = self.epochs,
            batch_size = self.batch_size
        )
    
    def predict(self, sentence):
        value = self.model.predict(self.dataModelation.transform([sentence]))
        return np.argmax(value[0])