from readWriteData import readData
from dataNormalization import BagOfWord, Vocabulary
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import json

class MlModel:
    def __init__(self):
        with open('config.json','r') as file:
            configs = json.load(file)
            max_features = configs['max_features']
            self.batch_size = configs['batch_size']
            norm = configs['norm']
            self.epochs = configs['epochs']
            vocabulary = configs['vocabulary']
            self.brains = configs['brains']
            if self.brains <= 0:
                self.brains = 1
            if self.brains & 1 == 0:
                # mereu vreau numar impar de creierase
                self.brains += 1

        data = None
        data = readData()
        
        if not vocabulary:
            self.dataModelation = BagOfWord(data, norm, max_features)
        else:
            self.dataModelation = Vocabulary(data,norm,max_features)

        self.generateDatasets()
        self.generateModel()
        self.trainModel() 
       
    def getCreatedModel(self):
        model = Sequential()
        model.add(Input(shape=(self.dataModelation.n), batch_size=self.batch_size))
        model.add(Dense(100, activation="relu"))
        model.add( Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def generateDatasets(self):
        n = int(len(self.dataModelation.labels) * 0.8)
        self.trainX = self.dataModelation.matrix[:n]
        self.trainY = self.dataModelation.labels[:n]
        self.testX = self.dataModelation.matrix[n:]
        self.testY = self.dataModelation.labels[n:]

    def generateModel(self): 
        self.model = [self.getCreatedModel() for _ in range(self.brains)]
    
    def trainModel(self):
        for model in self.model:
            model.fit( 
                self.trainX,
                to_categorical(self.trainY,2),
                epochs = self.epochs,
                batch_size = self.batch_size
            )

    def predict(self, value):
        arr = [
            np.argmax(
                model.predict(
                    self.dataModelation.transform([value])
                    )
                ) for model in self.model]

        d = [0,0]
        for i in arr:
            d[i] += 1

        if d[0] > d[1]:
            return 0

        return 1
        


    def testModel(self):
        predictions = [[np.argmax(i) for i in model.predict(self.testX)] for model in self.model]
        
        n = len(self.testY)
        d = [[0,0] for i in range(n)]

        for i in range(n):
            for j in range(self.brains):
                d[i][predictions[j][i]] += 1

        predictions = []
        for val in d:
            if val[0] > val[1]:
                predictions.append(0)
            else:
                predictions.append(1)

        print(classification_report(self.testY, predictions))


if __name__ == '__main__':
    model = MlModel()
    model.testModel()