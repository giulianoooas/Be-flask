from sklearn.feature_extraction.text import TfidfVectorizer
from dataProcessing import  textTokenize, textProcessing
from sklearn.preprocessing import normalize

class BagOfWord:
    
    def __init__(self,data, norm, max_features):
        """
            l1, l2 
        """
        self.bagOfWord = TfidfVectorizer(
            preprocessor=textProcessing,
            tokenizer = textTokenize,
            ngram_range=(1,2),
            token_pattern = None,
            max_features=max_features,
            norm=norm
        )
        self.labels = [val[1] for val in data]
        self.data  = [val[0] for val in data]
        self.bagOfWord.fit(self.data)
        self.matrix = self.bagOfWord.transform(self.data).toarray()
        self.n = self.matrix.shape[1]
        

    def transform(self, sentence):
        return self.bagOfWord.transform(sentence).toarray()
        

    
class Vocabulary:

    def __init__(self,data,norm,max_features):

        sentences = [textProcessing(i[0]) for i in data]
        self.data = [textTokenize(i) for i in sentences]
        self.labels = [i[1] for i in data]

        self.__d = {}
        self.max_featrues = max_features
        self.norm = norm
        self.n = 0
        self.generateDic()
        self.generateMatrix()
    
    def generateDic(self):
        n = 0
        for sentence in self.data:
            self.n = max(self.n,len(sentence))
            for token in sentence:
                if not self.__d.get(token):
                    self.__d[token] = n
                    n += 1

    def generateMatrix(self): 
        self.matrix = [self.transform(i,False) for i in self.data]
        self.matrix = normalize(self.matrix, norm = self.norm)

    def transform(self, sentence, takeFirst = True):
        if takeFirst:
            sentence = sentence[0]
        vector = []
        for token in sentence:
            vector.append(self.__d.get(token,0))
        vector = vector[:self.n]
        n = self.n - len(vector)
        res = vector + [1 for _ in range(n)] 
        if takeFirst:
            return [res]
        return res