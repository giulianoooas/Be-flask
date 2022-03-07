from re import I
from sklearn.feature_extraction import DictVectorizer

class DataModelation:
    
    def __init__(self,data):
        self.data = [d[0] for d in data]
        self.labels = [d[1] for d in data]
        self.voc = []
        self.linkVoc= {}
        self.n = 0
        self.bagOfWord()

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
