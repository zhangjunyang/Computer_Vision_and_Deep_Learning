import pandas as pd
from sklearn.ensemble import RandomForestRegressor
 
train = pd.read_csv( 'train.csv') #读取train数据
train_y = train.SalePrice
predictor_x = ['LotArea','YearBuilt','OverallQual','1stFlrSF','FullBath'] #特征
train_x = train[predictor_x]
my_model = RandomForestRegressor() #随机森林模型
 
my_model.fit(train_x,train_y) #fit
test = pd.read_csv( 'test.csv') #读取test数据
test_x = test[predictor_x]
pre_test_y = my_model.predict(test_x)
print(pre_test_y)
 
my_submission = pd.DataFrame({'Id':test.Id, 'SalePrice':pre_test_y}) #建csv
my_submission.to_csv('submission.csv', index=False)



