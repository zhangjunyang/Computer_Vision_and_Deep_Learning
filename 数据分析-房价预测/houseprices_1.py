import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv( '../input/train.csv') #读取train数据
train_y = train.SalePrice
predictor_x = ['LotArea','YearBuilt','OverallQual','1stFlrSF','FullBath'] #选取五个特征
train_x = train[predictor_x]

model = RandomForestRegressor() #随机森林模型
model.fit(train_x,train_y) #fit

test = pd.read_csv( '../input/test.csv') #读取test数据
test_x = test[predictor_x]

pre_test_y = model.predict(test_x)
print(pre_test_y)
 
my_submission = pd.DataFrame({'Id':test.Id, 'SalePrice':pre_test_y}) #建csv
my_submission.to_csv('submission.csv', index=False)


