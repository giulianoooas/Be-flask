import json, csv


def writeData(fileName, n):
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
    dic = [{'data': i[0], "label": i[1]} for i in data]
    with open('data.json','w') as writer:
        json.dump(dic,writer)

def readData():
    with open('data.json', 'r') as reader:
        data = json.load(reader)
    
    zeroData = [i for i in data if i['label'] == 0]
    oneData = [i for i in data if i['label'] == 1]

    res = []
    for i,j in zip(zeroData,oneData):
        res.extend([[i['data'],i['label']],[j['data'],j['label']]])

    return res

if __name__ == '__main__':
    N = 15000
    writeData('dataset.csv',N)
    data = readData()
    print(len(data))
    print(data[0])