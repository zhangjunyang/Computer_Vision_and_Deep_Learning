# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 

# load data files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def lotShape(x):

    # Function that converts categorical lot sizes to numeric
    if x == "Reg":
        return(3)
    if x == "IR1":
        return(2)
    if x == "IR2":
        return(1)
    return(0)

# Data Exploration
# obtain numeric variables
num_data_cols = list(train.dtypes[(train.dtypes == np.int64) | (train.dtypes == np.float64)][1:].index)

# plot different figures
plt.subplot(3, 4, 1)
plt.scatter(np.log(train.LotArea), np.log(train.SalePrice),marker='.')
plt.title('Lot Area vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 2)
plt.scatter(train.GarageArea, np.log(train.SalePrice),marker='.')
plt.title('Garage Area vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 3)
plt.scatter(np.log(train.LotFrontage), np.log(train.SalePrice),marker='.')
plt.title('Lot Frontage vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 4)
plt.scatter(train.YearBuilt, np.log(train.SalePrice),marker='.')
plt.title('Year built vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 5)
plt.scatter(train.OverallQual, np.log(train.SalePrice),marker='.')
plt.title('Overall condition vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 6)
plt.scatter(train.OverallCond, np.log(train.SalePrice),marker='.')
plt.title('Overall quality vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 7)
plt.scatter(np.log(train.GrLivArea), np.log(train.SalePrice),marker='.')
plt.title('Living Area vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 8)
plt.scatter(train.TotRmsAbvGrd, np.log(train.SalePrice),marker='.')
plt.title('Rooms vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 9)
plt.scatter(train.FullBath + train.HalfBath, np.log(train.SalePrice),marker='.')
plt.title('Baths vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 10)
plt.scatter(train.MSSubClass, np.log(train.SalePrice),marker='.')
plt.title('Class vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 11)
plt.scatter(train.YrSold, np.log(train.SalePrice),marker='.')
plt.title('Year sold vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(3, 4, 12)
plt.scatter(train.MoSold, np.log(train.SalePrice),marker='.')
plt.title('Month sold vs SalePrice')
plt.ylabel('log SalePrice')

plt.show()

# using seaborn
train['logLotArea'] = np.log(train.LotArea)
train['logGrLivArea'] = np.log(train.GrLivArea)
train['logSalePrice'] = np.log(train.SalePrice)
cols = ['logLotArea','GarageArea','OverallQual','OverallCond','logGrLivArea',]
sns.pairplot(train[cols])
plt.show()

# Feature Engineering
# create new data sets
train_fe = pd.DataFrame(data={"id": train.Id, "logSalePrice": np.log(train.SalePrice)})
test_fe = pd.DataFrame(data={"id": test.Id})

# take log lot sizes
train_fe['logLotArea'] = np.log(train.LotArea)
train_fe['logGrLivArea'] = np.log(train.GrLivArea)
test_fe['logLotArea'] = np.log(test.LotArea)
test_fe['logGrLivArea'] = np.log(test.GrLivArea)

# get number of NA values per column and delete features based on that
train.isnull().sum()
test.isnull().sum()

# FE Lot Frontage
# dont use
# FE MSZoning
MSZoning_values = list(train.MSZoning.unique())
for value in MSZoning_values:
    if value != 'C (all)':
        train_fe['MSZoning'+value] = ((train.MSZoning == value) | (train.MSZoning == 'C (all)')).astype(int)
        test_fe['MSZoning'+value] = ((test.MSZoning == value) | (test.MSZoning == 'C (all)')).astype(int)

# check explanatory power of MSZoning dummies
plt.subplot(2, 2, 1)
plt.scatter(train_fe.MSZoningRL, train_fe.logSalePrice,marker='.')
plt.title('MSZoning RL vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(2, 2, 2)
plt.scatter(train_fe.MSZoningRM, train_fe.logSalePrice,marker='.')
plt.title('MSZoning RM vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(2, 2, 3)
plt.scatter(train_fe.MSZoningFV, train_fe.logSalePrice,marker='.')
plt.title('MSZoning FV vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(2, 2, 4)
plt.scatter(train_fe.MSZoningRH, train_fe.logSalePrice,marker='.')
plt.title('MSZoning RH vs SalePrice')
plt.ylabel('log SalePrice')

# corr plot
cm = np.corrcoef(train_fe[['MSZoningRL','MSZoningRM','MSZoningFV','MSZoningRH','logSalePrice']].values.T) 
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=['MSZoningRL','MSZoningRM','MSZoningFV','MSZoningRH','logSalePrice'],
                 xticklabels=['MSZoningRL','MSZoningRM','MSZoningFV','MSZoningRH','logSalePrice'])


# FE Street
train_fe['isStreetPaved'] = (train.Street == "Pave").astype(int)
test_fe['isStreetPaved'] = (test.Street == "Pave").astype(int)

# almost all observations have paved street
1-sum(train_fe['isStreetPaved'])/train_fe.shape[0] # don't use

# drop columns
train_fe.drop('isStreetPaved',axis=1,inplace=True)
test_fe.drop('isStreetPaved',axis=1,inplace=True)


# FE Alley
train_fe['AlleyGrvl'] = (train.Alley == "Grvl").astype(int)
train_fe['AlleyPave'] = (train.Alley == "Pave").astype(int)
test_fe['AlleyGrvl'] = (test.Alley == "Grvl").astype(int)
test_fe['AlleyPave'] = (test.Alley == "Pave").astype(int)

cm = np.corrcoef(train_fe[['AlleyGrvl','AlleyPave']].values.T)
sum(train_fe['AlleyGrvl'])/train_fe.shape[0]
sum(train_fe['AlleyPave'])/train_fe.shape[0]

# FE Lot Shape
train_fe['LotShape'] = train['LotShape'].apply(lambda x: lotShape(x))
test_fe['LotShape'] = test['LotShape'].apply(lambda x: lotShape(x))

plt.scatter(train_fe.LotShape, train_fe.logSalePrice, marker='.')

# FE Land Contour
for value in list(train['LandContour'].unique()):
    train_fe["LandContour"+value] = (train['LandContour'] == value).astype(int)
    test_fe["LandContour"+value] = (test['LandContour'] == value).astype(int)
del train_fe["LandContourLvl"]
del test_fe["LandContourLvl"]

plt.subplot(2, 2, 1)
plt.scatter(train_fe.LandContourBnk, train_fe.logSalePrice,marker='.')
plt.title('LandContour Bnk vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(2, 2, 2)
plt.scatter(train_fe.LandContourLow, train_fe.logSalePrice,marker='.')
plt.title('LandContour Low vs SalePrice')
plt.ylabel('log SalePrice')

plt.subplot(2, 2, 3)
plt.scatter(train_fe.LandContourHLS, train_fe.logSalePrice,marker='.')
plt.title('LandContour HLS vs SalePrice')
plt.ylabel('log SalePrice')


