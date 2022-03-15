from sklearn.feature_extraction.text import TfidfVectorizer
from dataProcessing import  textTokenize, textProcessing

class DataModelation:
    
    def __init__(self,data, type, max_features):
        """
            l1, l2 
        """
        self.bagOfWord = TfidfVectorizer(
            preprocessor=textProcessing,
            tokenizer = textTokenize,
            ngram_range=(1,2),
            token_pattern = None,
            max_features=max_features,
            norm=type
        )
        self.labels = [val[1] for val in data]
        self.data  = [val[0] for val in data]
        self.bagOfWord.fit(self.data)
        self.matrix = self.bagOfWord.transform(self.data).toarray()
        

    def transform(self, sentence):
        return self.bagOfWord.transform(sentence).toarray()
        

    
