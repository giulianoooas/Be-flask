from dataProcessing import getGoodData, textProccessing
from dataNormalization import DataModelation

data = getGoodData(200)
dataModelation = DataModelation(data, 'L1')