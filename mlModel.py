from dataProcessing import getGoodData
from dataNormalization import BagOfWord, Vocabulary
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from tensorflow.keras.utils import to_categorical
import json

class NeuralNetworkModel:
    def __init__(self):
        with open('config.json','r') as file:
            configs = json.load(file)
            max_features = configs['max_features']
            n = configs['n']
            self.batch_size = configs['batch_size']
            norm = configs['norm']
            self.epochs = configs['epochs']
            vocabulary = configs['vocabulary']
            self.max_length = configs['max_length']

        data = None
        data = getGoodData(n)
        
        if not vocabulary:
            self.dataModelation = BagOfWord(data, norm, max_features)
        else:
            self.dataModelation = Vocabulary(data,norm,max_features,self.max_length)
        self.generateModel()
        self.trainModel()
       

    
    def generateModel(self): 
        n = self.max_length
        try:
            n = self.dataModelation.matrix.shape[1]
        except:
            pass
        self.model = Sequential()
        self.model.add(Input(shape=(n), batch_size=self.batch_size))
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