import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, preprocessing
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore') 

label_encoder = preprocessing.LabelEncoder();

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


df_train.info()
df_train.corr()

#plt.scatter(df_train.LotArea, df_train.SalePrice)
#plt.show()

#let's see missing value count group by each variable and the % of missing for each variable
pd.concat([df_train.isnull().sum(), df_train.isnull().sum()/df_train.isnull().count()*100], axis=1)

def getModeForGarageQual(x):
       return df_train[df_train.GarageYrBlt==x].GarageQual.mode()

def fill_missing_lotfrontage(x):
       data_subset , alldata = x;
       ##get mean(LotFrontate/LotArea ratio) and multiply by LotArea to get the missing lotfrontage
       return data_subset.LotArea*np.mean(alldata[pd.notnull(alldata.LotFrontage)].LotFrontage/alldata[pd.notnull(alldata.LotFrontage)].LotArea);

#TotalBsmtSF has also missing values hence call this function after we fill that;
def fill_missing_BsmtUnfSF(x):
       data_subset , alldata = x;
       ##get mean(BsmtUnfSF/TotalBsmtSF ratio) and multiply by TotalBsmtSF to get the missing lotfrontage
       return data_subset.TotalBsmtSF*np.mean(alldata[pd.notnull(alldata.BsmtUnfSF)].BsmtUnfSF/alldata[pd.notnull(alldata.BsmtUnfSF)].TotalBsmtSF);

#function to prepend column values with the column name
def update_qualitative_col_values(x):
       dataset, col_name = x
       return str(col_name)+"-"+ str(dataset[col_name])





#Missing value - let's get rid of the ones that don't make much sense;
cols=['Id', 'MSSubClass', 'MSZoning',  'LotArea',
       'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF',
       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional',
       'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
       'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice']

col_qualitative=['HeatingQC', 'KitchenQual','BsmtQual','Foundation','ExterQual','GarageFinish','HouseStyle','BsmtFinType1'];

#update qualitative column data
for col_name in col_qualitative:
       df_train[col_name]=df_train.apply(lambda x: update_qualitative_col_values([x, col_name]),axis=1);

for col_name in col_qualitative:
       df_test[col_name]=df_test.apply(lambda x: update_qualitative_col_values([x, col_name]),axis=1);


##'GarageArea' ---Kaggle score went down but mean squared error reduced


GarageYrBlt_mode = df_train.GarageYrBlt.mode()[0];
df_train.loc[df_train.GarageYrBlt.isnull(),'GarageYrBlt']=GarageYrBlt_mode;
df_train.loc[df_train.MasVnrArea.isnull(),'MasVnrArea']=df_train.MasVnrArea.median()
df_train.loc[df_train.GarageCars.isnull(),'GarageCars']=df_train.GarageCars.median()
df_train.loc[df_train.GarageArea.isnull(),'GarageArea']=df_train.GarageArea.median()
df_train.loc[df_train.TotalBsmtSF.isnull(),'TotalBsmtSF']=df_train.TotalBsmtSF.median()
df_train.loc[df_train.BsmtFinSF1.isnull(),'BsmtFinSF1']=df_train.BsmtFinSF1.median()
df_train.loc[df_train.GarageQual.isnull(),['GarageQual']]=df_train.GarageQual.mode()[0]


# Saleprice does not follow normal distribution; positive skewness; log transformation works great for positive skewness;
# this helped get better score/R squared; also mean squared error reduced drastically; also importe rank in kaggle
df_train['SalePriceLOG'] = np.log(df_train['SalePrice']) 
#check for skewness; see sns.distplot histogram
df_train['LotAreaLOG'] = np.log(df_train['LotArea']) # LotArea does not follow normal distribution; positive skewness; log transformation works great for positive skewness;
#check for skewness; see sns.distplot histogram

# OpenPorchSF does not follow normal distribution; positive skewness; log transformation works great for positive skewness;
#df_train['OpenPorchSF_LOG'] = np.log1p(df_train['OpenPorchSF'])

df_train.loc[pd.isnull(df_train['LotFrontage']),'LotFrontage'] = df_train[pd.isnull(df_train['LotFrontage'])].apply((lambda x: fill_missing_lotfrontage([x, df_train])),axis=1)
df_train.loc[pd.isnull(df_train['BsmtUnfSF']),'BsmtUnfSF'] = df_train[pd.isnull(df_train['BsmtUnfSF'])].apply((lambda x: fill_missing_BsmtUnfSF([x, df_train])),axis=1)

#to address Additive assumption associated with Linear regression
df_train['yrsold'] = 2017-df_train.YearBuilt
df_train['yrsoldQual'] = df_train.OverallQual*df_train.yrsold

#when we see the corr() for following 2 predictors we can see there is some relationship between them;
#hence when a new variable is created combining these two variables and include the new variable in the prediction, we see there is an impact on MSE
#surprisingly MSE on training data increased but model did better in kaggle LB
df_train['LotFrontage-WoodDeckSF']=df_train.LotFrontage*df_train.WoodDeckSF


#df_train['BsmtHalfBath-FullBath']= df_train.FullBath+df_train.HalfBath --local MSE goes down but kaggle LB score goes up

#when we see the corr() for following 2 predictors we can see there is some relationship between them;
#hence when a new variable is created combining these two variables and include the new variable in the prediction, we see there is an impact on MSE; local MSE went up;
#surprisingly MSE on training data increased but model did better in kaggle LB
#rationale for this can be assumed as house with more bedrooms would need more bathroom; so more bathroom with more number of bathrooms will have better value;
df_train['BedrookAbvGr-FullBath']=df_train.BedroomAbvGr*df_train.FullBath


#create a new variable from yearbuilt to find out how many years old the property is; we will get a new quantitative variable;
#if you check corr, it's >52% inverse which is great



GarageYrBlt_mode = df_test.GarageYrBlt.mode()[0];
df_test.loc[df_test.GarageYrBlt.isnull(),'GarageYrBlt']=GarageYrBlt_mode;
df_test.loc[df_test.MasVnrArea.isnull(),'MasVnrArea']=df_test.MasVnrArea.median()
df_test.loc[df_test.GarageCars.isnull(),'GarageCars']=df_test.GarageCars.median()
df_test.loc[df_test.GarageArea.isnull(),'GarageArea']=df_test.GarageArea.median()
df_test.loc[df_test.TotalBsmtSF.isnull(),'TotalBsmtSF']=df_test.TotalBsmtSF.median()
df_test.loc[df_test.BsmtFinSF1.isnull(),'BsmtFinSF1']=df_test.BsmtFinSF1.median()
df_test.loc[df_test.GarageQual.isnull(),['GarageQual']]=df_test.GarageQual.mode()[0]
df_test['LotAreaLOG'] = np.log(df_test['LotArea']) # LotArea does not follow normal distribution; positive skewness; log transformation works great for positive skewness;
##df_test['OpenPorchSF_LOG'] = np.log1p(df_test['OpenPorchSF'])  ##no improment in kaggle; although mean squared error reduced;

df_test.loc[pd.isnull(df_test['LotFrontage']),'LotFrontage'] = df_test[pd.isnull(df_test['LotFrontage'])].apply((lambda x: fill_missing_lotfrontage([x, df_test])),axis=1)
df_test.loc[pd.isnull(df_test['BsmtUnfSF']),'BsmtUnfSF'] = df_test[pd.isnull(df_test['BsmtUnfSF'])].apply((lambda x: fill_missing_BsmtUnfSF([x, df_test])),axis=1)

#to address Additive assumption associated with Linear regression
df_test['yrsold'] = 2017-df_test.YearBuilt
df_test['yrsoldQual'] = df_test.OverallQual*df_test.yrsold


df_test['LotFrontage-WoodDeckSF']=df_test.LotFrontage*df_test.WoodDeckSF

#df_test['BsmtHalfBath-FullBath']= df_test.FullBath+df_test.HalfBath
df_test['BedrookAbvGr-FullBath']=df_test.BedroomAbvGr*df_test.FullBath

#########training data manupulation; new feature addition; feature removal#############

df_train = df_train[df_train.LotArea<60000]

#encode qualitative variables
df_train['StreetEn'] = label_encoder.fit_transform(df_train["Street"])
df_train['MSZoningEn'] = label_encoder.fit_transform(df_train["MSZoning"])
df_train['GarageQualEn'] = label_encoder.fit_transform(df_train["GarageQual"])
df_train['LotShapeEn'] = label_encoder.fit_transform(df_train["LotShape"])
df_train['UtilitiesEn'] = label_encoder.fit_transform(df_train["Utilities"])
df_train['GrLivAreaSQ']= df_train.GrLivArea**2;


#Qualitative features; integer; no missing data;
df_train.Street.unique()

#Qualitative features; no missing data;
df_train.MSSubClass.unique()
df_train.MSZoning.unique()

# data exploration
#sns.boxplot(df_trian.YearBuilt, df_trian.SalePrice)
#plt.show()
#MoSold -- categorical; let's see if houseprice has something to do with the month they are sold; i think so; no correlation
#YrSold --no use; not much impact on corr on SalePrice

##data exploration###
#LotShape; check boxplot 
#sns.boxplot(df_train.)

#WoodDeckSF; check skewness, distplot, corr, scatter; this is quantitative

#BsmtUnfSF --check skewness; distplot, corr, scatter; log1p and squared might help; this is quantitative

#test data manupulation; new feature addition; feature removal; address missing data#############

#encode qualitative variables
df_test['StreetEn'] = label_encoder.fit_transform(df_test["Street"])
df_test['MSZoningEn'] = label_encoder.fit_transform(df_test["MSZoning"])
df_test['GarageQualEn'] = label_encoder.fit_transform(df_test["GarageQual"])
df_test['LotShapeEn'] = label_encoder.fit_transform(df_test["LotShape"])
df_test['UtilitiesEn'] = label_encoder.fit_transform(df_test["Utilities"]) ##no improvement in kaggle; although mean square error came down
df_test['GrLivAreaSQ']= df_test.GrLivArea**2;

#add columns that need to be part of the model
#rules-
#1. features that have high correlation with SalePrice, just add them;
#following have corr > .4
#OverallQual YearBuilt YearRemodAdd MasVnrArea BsmtFinSF1 TotalBsmtSF 1stFlrSF GrLivArea FullBath TotRmsAbvGrd Fireplaces GarageYrBlt GarageCars GarageArea
#let's add most corr features
#df_train.corr()[abs(df_train.corr().SalePrice)>.4].SalePrice 

df_train=pd.get_dummies(df_train, prefix="col")
df_test=pd.get_dummies(df_test, prefix="col")

most_effective_dummy_cols = df_train.corr()[df_train.corr().SalePrice>.35].index.tolist() #that has more than .35 in corr

#pd.DataFrame(pd.concat([df_train.col_Fin, df_train.SalePrice],ignore_index=False, axis=1)).corr() --get corr between two variables

#col_NoSeWa - no improvement
#, 'col_Foundation-PConc' --Local score improved bu Kaggle LB went down
#col_ExterQual-Ex --not good
#'col_ExterQual-Gd' --Local score improved bu Kaggle LB went down
#,'HalfBath',  --Local score improves but no impact on Kaggle LB ; also P-value is relatively larger for this predictor

cols_test=['LotAreaLOG','PoolArea', 'StreetEn', 'MSSubClass', 'MSZoningEn', 'OverallQual','YearBuilt','YearRemodAdd'
,'1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','Fireplaces','MasVnrArea','GarageCars','TotalBsmtSF', 'BsmtFinSF1'
,'GarageQualEn', 'GrLivAreaSQ'
,'LotFrontage','LotShapeEn','WoodDeckSF','BsmtUnfSF'
,'yrsoldQual','LotFrontage-WoodDeckSF','BedrookAbvGr-FullBath'
,'col_FR2', 'col_New','col_WD', 'col_Normal','col_HeatingQC-Ex','col_KitchenQual-Ex','col_BsmtQual-Ex','col_HouseStyle-2Story','col_GarageFinish-Fin'
,'col_BsmtFinType1-GLQ','col_BsmtFinType1-ALQ','col_BsmtFinType1-Unf','col_BsmtFinType1-Rec','col_BsmtFinType1-BLQ','col_BsmtFinType1-LwQ'
]

#cols_test = cols_test + most_effective_dummy_cols

data_X_train=df_train[cols_test]
data_y_train=df_train.SalePriceLOG

#compute some key stats to prove H0 hypothesis is not applicable; if F-stat >1 and p-value is close to 0 then H0 can be rejected;
results = sm.OLS(data_y_train, data_X_train).fit();
results.summary()

data_X_test = df_test[cols_test]

#data transfromation to higher degree polynomial
# poly = preprocessing.PolynomialFeatures(2) #Local score and MSE improves but kaggle LB score does not improve; it's trying to overfit;
# data_X_train = poly.fit_transform(data_X_train) #Local score and MSE improves but kaggle LB score does not improve;
# data_X_test = poly.fit_transform(data_X_test) #Local score and MSE improves but kaggle LB score does not improve;


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_X_train, data_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)

#The mean squared error; should be as low as possible and if it's reducing then we are going in right direction otherwise not;
print('Mean squared error \n', np.mean((np.exp(regr.predict(data_X_train))-df_train.SalePrice)**2))

# 1 is perfect i.e. 100% explained; i.e. change in saleprice due to any change in any features can be fully explained by the model;; this is nothing but R squared;
print('Variance score:%.2f' % regr.score(data_X_train,data_y_train))

##predict and create file for Kaggle submission

houseprices_test = regr.predict(data_X_test)
submission = pd.DataFrame({'Id':df_test.Id, 'SalePrice':np.exp((houseprices_test))})
submission.to_csv('submission3.csv', index=False)