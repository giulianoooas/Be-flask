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
                if not self.linkVoc.get(word1):
                    self.linkVoc[word1] = self.n
                    self.voc.append(word1)
                    self.n += 1
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
                arr[self.linkVoc.get(word)] += 1
                if word1:
                   arr[self.linkVoc.get(word1)] += 1 
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
        n = len(tokens)
        for i in range(n):
            token = tokens[i]
            word1 = None
            if i + 1 < n:
                word1 = token + ' ' + tokens[i + 1]
            if self.linkVoc.get(token):
                arr[self.linkVoc.get(token)] += 1
            if self.linkVoc.get(word1):
                arr[self.linkVoc.get(word1)] += 1
        
        if type == 'S':
            return self.scaler.transform(arr)
        elif type == 'L1':
            return sklearn.preprocessing.normalize(arr, norm='l1')
        else:
            return sklearn.preprocessing.normalize(arr, norm='l1')

    
