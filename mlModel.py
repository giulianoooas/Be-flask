from readWriteData import readData
from dataNormalization import BagOfWord, Vocabulary
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import json

class NeuralNetworkModel:
    def __init__(self):
        with open('config.json','r') as file:
            configs = json.load(file)
            max_features = configs['max_features']
            self.batch_size = configs['batch_size']
            norm = configs['norm']
            self.epochs = configs['epochs']
            vocabulary = configs['vocabulary']

        data = None
        data = readData()
        
        if not vocabulary:
            self.dataModelation = BagOfWord(data, norm, max_features)
        else:
            self.dataModelation = Vocabulary(data,norm,max_features)

        self.generateDatasets()
        self.generateModel()
        self.trainModel() 
       

    def generateDatasets(self):
        n = int(len(self.dataModelation.labels) * 0.8)
        self.trainX = self.dataModelation.matrix[:n]
        self.trainY = self.dataModelation.labels[:n]
        self.testX = self.dataModelation.matrix[n:]
        self.testY = self.dataModelation.labels[n:]

    def generateModel(self): 
        self.model = Sequential()
        self.model.add(Input(shape=(self.dataModelation.n), batch_size=self.batch_size))
        self.model.add(Dense(64, activation="relu"))
        self.model.add( Dense(64, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(2, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def trainModel(self):
        self.model.fit( 
            self.trainX,
            to_categorical(self.trainY,2),
            epochs = self.epochs,
            batch_size = self.batch_size
        )
    
    def testModel(self):
        predictions = [np.argmax(i) for i in self.model.predict(self.testX)]
        print(classification_report(self.testY, predictions))

    def predict(self, sentence):
        value = self.model.predict(self.dataModelation.transform([sentence]))
        return np.argmax(value[0])

if __name__ == '__main__':
    model = NeuralNetworkModel()
    model.testModel()