import numpy as np
import sklearn
from dataProcessing import textProccessing
import json
from db import DB

class DataModelation:
    
    def __init__(self,data, type, reload, max_features):
        """
            S -> Standardization
            L1, L2 
        """
        self.type = type
        self.db = DB()
        self.max_features = max_features

        if reload:
            self.data = [d[0] for d in data]
            self.labels =[d[1] for d in data] 
            self.voc = []
            self.linkVoc= {}
            self.n = 0
            self.bagOfWord()
        else:
            self.loadDataFromDB()
               
        if type == 'S':
                self.Standardization()
        elif type == 'L1':
            self.L1()
        else:
            self.L2()
    
    def loadDataFromDB(self):
        data = self.db.getModel()
        data = json.loads(data[0])
        self.matrix = np.array(data['matrix'])
        self.labels = np.array(data['labels'])
        self.voc = data['voc']
        self.linkVoc = data['linkVoc']
        self.n = data['n']



    def bagOfWord(self):
        for sentence in self.data:
            n = len(sentence)
            for i in range(n):
                word = sentence[i]
                word1 = word
                if i + 1 < n:
                    word1 = word + ' ' + sentence[i + 1]
                if not self.linkVoc.get(word):
                    self.linkVoc[word] = self.n
                    self.voc.append(word)
                    self.n += 1
                    if self.n == self.max_features:
                        break
                if not self.linkVoc.get(word1):
                    self.linkVoc[word1] = self.n
                    self.voc.append(word1)
                    self.n += 1
                    if self.n == self.max_features:
                        break
        self.generateMatrix()
    
    def generateMatrix(self):
        self.matrix = []
        for sentence in self.data:
            n = len(sentence)
            arr = [0 for i in range(self.n)]
            for i in range(n):
                word = sentence[i]
                word1 = None
                if i + 1 < n:
                    word1 = word + ' ' + sentence[i + 1]
                if self.linkVoc.get(word):
                    arr[self.linkVoc.get(word)] += 1
                if word1 and self.linkVoc.get(word1):
                   arr[self.linkVoc.get(word1)] += 1 
            self.matrix.append(arr)            

        self.matrix = np.array(self.matrix)
        self.labels = np.array(self.labels)
    
    def L1(self):
        self.matrix = sklearn.preprocessing.normalize(self.matrix, norm='l1')
        self.saveInDb()
        
    
    def L2(self):
        self.matrix = sklearn.preprocessing.normalize(self.matrix, norm='l2')
        self.saveInDb()

    def saveInDb(self):
        data = json.dumps({
            'n': self.n,
            'matrix': self.matrix.tolist(),
            'labels': self.labels.tolist(),
            'voc': self.voc,
            'linkVoc': self.linkVoc
        })
        self.db.addInDb(data)


    def Standardization(self):
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(self.matrix)
        self.matrix = self.scaler.transform(self.matrix)
        self.saveInDb()

    def transform(self, sentence):
        tokens = textProccessing([[sentence, 0]])[0][0]
        arr = np.zeros(shape=(self.n,))
        ok = False
        n = len(tokens)
        for i in range(n):
            token = tokens[i]
            word1 = None
            if i + 1 < n:
                word1 = token + ' ' + tokens[i + 1]
            if self.linkVoc.get(token):
                arr[self.linkVoc.get(token)] += 1
                ok = True
            if self.linkVoc.get(word1):
                ok = True
                arr[self.linkVoc.get(word1)] += 1
        
        if not ok:
            return np.array([arr])
        if type == 'S':
            return self.scaler.transform([arr])
        elif type == 'L1':
            return sklearn.preprocessing.normalize([arr], norm='l1')
        else:
            return sklearn.preprocessing.normalize([arr], norm='l1')

    
