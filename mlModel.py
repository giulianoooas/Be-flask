from dataProcessing import getGoodData
from dataNormalization import DataModelation

data = getGoodData(200)
dataModelation = DataModelation(data, 'S')
print(dataModelation.test('am ad sd aa'))