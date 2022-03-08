#kagle dataset https://www.kaggle.com/devkhant24/toxic-comment/version/4?select=jigsaw-toxic-comment-train.csv
# labels : 0 -> bad, 1 = good
import csv, re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords


stemmer = SnowballStemmer(language="english")
regexUrl = r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
regexPunctuation = r'[^a-z]'

#partea de procesare de date

def getData(fileName):
    data = []
    with open(fileName, 'r', encoding="utf8") as file:
        rows = csv.reader(file)
        next(rows)
        for row in rows:
            label = 1
            if row[2:].count('1'):
                label = 0
            data.append([row[1], label])
    return data

def removeStopWords(data):
    nltk.download('stopwords')
    stop_words_nltk = set(stopwords.words('english'))
    newData = []
    for row in data:
        new = []
        for word in row[0]:
            if not word in stop_words_nltk:
                new.append(word)
        newData.append([new,row[1]])
    return newData

    

def makeLower(data):
    newData = []
    for row in data:
        newData.append([row[0].lower(), row[1]])
    return newData

def removeRegex(data,regex):
    global ok
    newData= []
    for row in data:
        a = row[0]
        matches = re.findall(regex,row[0])
        for match in matches:
            if match != '':
                a = a.replace(match,' ')
        newData.append([a,row[1]])
    return newData

def tokenized(data):
    newData = []
    for row in data:
        newData.append([word_tokenize(row[0]),row[1]])
    return newData

def stemming(data, stemmer):
    newData = []
    for row in data:
        arr = []
        for token in row[0]:
            arr.append(stemmer.stem(token))
        newData.append([arr,row[1]])
    return newData

def textProccessing(data):
    data = makeLower(data)
    data = removeRegex(data,regexUrl)
    data = removeRegex(data, regexPunctuation)
    data = tokenized(data)
    data = removeStopWords(data)
    data = stemming(data,stemmer)
    return data



def getGoodData(n):
    data = getData('dataset.csv')[:n]
    data = textProccessing(data)
    return data