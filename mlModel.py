from dataProcessing import getGoodData
from dataNormalization import DataModelation
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from tensorflow.keras.utils import to_categorical
import json

class NeuralNetworkModel:
    def __init__(self, n, normalization, batch_size, reload = False):
        data = None
        if reload:
            data = getGoodData(n)
            with open('json_data.json', 'w') as file:
                json.dump(data, file)
        else:
            with open('json_data.json', 'r') as file:
                data = json.load(file)

        self.dataModelation = DataModelation(data, normalization,reload)
        self.batch_size=batch_size
        self.generateModel()
        self.trainModel()
       

    
    def generateModel(self): 
        self.model = Sequential()
        self.model.add(Input(shape=(self.dataModelation.n), batch_size=self.batch_size))
        self.model.add(Dense(64, activation="relu"))
        self.model.add( Dense(16, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(2, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def trainModel(self):
        print(type(self.dataModelation.matrix))
        self.model.fit( 
            self.dataModelation.matrix,
            to_categorical(self.dataModelation.labels,2),
            epochs = 5,
            batch_size = self.batch_size
        )
    
    def predict(self, sentence):
        value = self.model.predict(self.dataModelation.transform(sentence))
        print(value)
        return np.argmax(value[0])

n = NeuralNetworkModel(100,'L2',100, False)