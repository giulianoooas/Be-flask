import numpy as np
from sklearn.feature_extraction import DictVectorizer
import sklearn
from dataProcessing import textProccessing

class DataModelation:
    
    def __init__(self,data, type="L2"):
        """
            S -> Standardization
            L1, L2 
        """
        self.data = [d[0] for d in data]
        self.labels = [d[1] for d in data]
        self.voc = []
        self.linkVoc= {}
        self.n = 0
        self.bagOfWord()
        self.type = type
        if type == 'S':
            self.Standardization()
        elif type == 'L1':
            self.L1()
        else:
            self.L2()

    def bagOfWord(self):
        for sentence in self.data:
            for word in sentence:
                if not self.linkVoc.get(word):
                    self.linkVoc[word] = self.n
                    self.voc.append(word)
                    self.n += 1
        self.generateMatrix()
    
    def generateMatrix(self):
        self.matrix = []
        for sentence in self.data:
            arr = [0 for i in range(self.n)]
            for word in sentence:
                arr[self.linkVoc.get(word)] += 1
            self.matrix.append(arr)
            
        self.matrix = np.array(self.matrix)
    
    def L1(self):
        self.matrix = sklearn.preprocessing.normalize(self.matrix, norm='l1')
    
    def L2(self):
        self.matrix = sklearn.preprocessing.normalize(self.matrix, norm='l2')


    def Standardization(self):
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(self.matrix)
        self.matrix = self.scaler.transform(self.matrix)

    def test(self, sentence):
        tokens = textProccessing([[sentence, 0]])[0][0]
        arr = [0 for i in range(self.n)]
        for token in tokens:
            if self.linkVoc.get(token):
                arr[self.linkVoc.get(token)] += 1
        
        if type == 'S':
            return self.scaler.transform(arr)
        elif type == 'L1':
            return sklearn.preprocessing.normalize(arr, norm='l1')
        else:
            return sklearn.preprocessing.normalize(arr, norm='l1')

    
