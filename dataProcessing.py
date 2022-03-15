#kagle dataset https://www.kaggle.com/devkhant24/toxic-comment/version/4?select=jigsaw-toxic-comment-train.csv
# labels : 0 -> bad, 1 = good
import csv, re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words_nltk = set(stopwords.words('english'))

stemmer = SnowballStemmer(language="english")
regexUrl = r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
regexPunctuation = r'[^a-z]'

#partea de procesare de date

def getData(fileName, n):
    nr0 = 0
    nr1 = 0
    data = []
    with open(fileName, 'r', encoding="utf8") as file:
        rows = csv.reader(file)
        next(rows)
        for row in rows:
            label = 1
            if row[2:].count('1'):
                label = 0
            if (nr0 < n and label == 0 ):
                data.append([row[1], label])
                nr0 += 1
            if (nr1 < n and label == 1):
                data.append([row[1], label])
                nr1 += 1
            
            if nr1 == n and nr0 == n:
                break
    return data

def removeStopWords(data):
    new = []
    for word in data:
        if not word in stop_words_nltk:
            new.append(word)
    return new
    

    

def makeLower(data):
    return data.lower()

def removeUrl(data):
    matches = re.findall(r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)',data)
    a = data
    for match in matches:
        if match != '':
            a = a.replace(match,' ')
    return data

def removeNoLetters(data):
    matches = re.findall(r'[^a-z]',data)
    a = data
    for match in matches:
        if match != '':
            a = a.replace(match,' ')
    return data


def stemming(data):
    return [stemmer.stem(token) for token in data]

def textProcessing(data):
    data = makeLower(data)
    data = removeUrl(data)
    data = removeNoLetters(data)
    return data

def textTokenize(data):
    data = word_tokenize(data)
    data = removeStopWords(data)
    data = stemming(data)
    return data

def getGoodData(n):
    data = getData('dataset.csv',n)
    return data