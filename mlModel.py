from dataProcessing import getGoodData
from dataNormalization import DataModelation
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

class NeuralNetworkModel:
    def __init__(self, n, normalization, batch_size):
        data = getGoodData(n)
        self.dataModelation = DataModelation(data, normalization)
        self.batch_size=batch_size
        self.generateModel()
        self.trainModel()
       

    
    def generateModel(self): 
        self.model = Sequential()
        self.model.add(Input(shape=(self.dataModelation.n), batch_size=self.batch_size))
        self.model.add(Dense(64, activation="relu"))
        self.model.add( Dense(16, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(2, activation="sigmoid"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def trainModel(self):
        self.model.fit( 
            self.dataModelation.matrix,
            self.dataModelation.labels,
            epochs = 50,
            batch_size = self.batch_size
        )
    
    def predict(self, sentence):
        return np.argmax(self.model.predict(self.dataModelation.transform(sentence)))

n = NeuralNetworkModel(200,'L2',20)
print(n.predict('How are you'))