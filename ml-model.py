#kagle dataset https://www.kaggle.com/devkhant24/toxic-comment/version/4?select=jigsaw-toxic-comment-train.csv
# labels : 0 -> bad, 1 = good
import csv, re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer(language="english")

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words

ok = True
regexUrl = r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
regexPunctuation = r'(!|@|#|$|%|^|&|\*|\(|\)|-|_|\+|=|\{|\}|\[|\]|\||\\|\n|\t|:|;|"|\'|<|,|\.|\?)?'

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


data = getData('dataset.csv')[:100]

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
        print(matches)
        if ok and len(matches):
            ok = False
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

def textProccessing(data):
    data = makeLower(data)
    data = removeRegex(data,regexUrl)
    data = removeRegex(data, regexPunctuation)
    data = tokenized(data)
    print(data)

data = textProccessing(data)