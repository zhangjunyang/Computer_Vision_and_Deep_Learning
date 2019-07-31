# https://www.kaggle.com/c/bike-sharing-demand/data

import pandas as pd
df = pd.read_csv('train.csv')
pd.set_option('display.max_rows',4 )
df

df.info()
df.describe()

for i in range(5, 12):
 name = df.columns[i]
 print('{0}偏态系数为 {1}, 峰态系数为 {2}'.format(name, df[name].skew(), df[name].kurt()))

print('未去重: ', df.shape)
print('去重: ', df.drop_duplicates().shape)

import seaborn as sns
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
#绘制箱线图
sns.boxplot(x="windspeed", data=df,ax=axes[0][0])
sns.boxplot(x='casual', data=df, ax=axes[0][1])
sns.boxplot(x='registered', data=df, ax=axes[1][0])
sns.boxplot(x='count', data=df, ax=axes[1][1])
plt.show()

#转换格式, 并提取出小时, 星期几, 月份
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df.datetime.dt.hour
df['week'] = df.datetime.dt.dayofweek
df['month'] = df.datetime.dt.month
df['year_month'] = df.datetime.dt.strftime('%Y-%m')
df['date'] = df.datetime.dt.date
#删除datetime
df.drop('datetime', axis = 1, inplace = True)
df

# 日期和总租赁数量
import matplotlib
#设置中文字体
font = {'family': 'SimHei'}
matplotlib.rc('font', **font)
#分别计算日期和月份中位数
group_date = df.groupby('date')['count'].median()
group_month = df.groupby('year_month')['count'].median()
group_month.index = pd.to_datetime(group_month.index)
plt.figure(figsize=(16,5))
plt.plot(group_date.index, group_date.values, '-', color = 'b', label = '每天租赁数量中位数', alpha=0.8)
plt.plot(group_month.index, group_month.values, '-o', color='orange', label = '每月租赁数量中位数')
plt.legend()
plt.show()

# 月份和总租赁数量
import seaborn as sns
plt.figure(figsize=(10, 4))
sns.boxplot(x='month', y='count', data=df)
plt.show()

# 季节和总租赁数量
plt.figure(figsize=(8, 4))
sns.boxplot(x='season', y='count', data=df)
plt.show()

# 星期几和租赁数量
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
sns.boxplot(x="week",y='casual' ,data=df,ax=axes[0])
sns.boxplot(x='week',y='registered', data=df, ax=axes[1])
sns.boxplot(x='week',y='count', data=df, ax=axes[2])
plt.show()

# 节假日, 工作日和总租赁数量
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 7))
sns.boxplot(x='holiday', y='casual', data=df, ax=axes[0][0])
sns.boxplot(x='holiday', y='registered', data=df, ax=axes[1][0])
sns.boxplot(x='holiday', y='count', data=df, ax=axes[2][0])
sns.boxplot(x='workingday', y='casual', data=df, ax=axes[0][1])
sns.boxplot(x='workingday', y='registered', data=df, ax=axes[1][1])
sns.boxplot(x='workingday', y='count', data=df, ax=axes[2][1])
plt.show()

# 小时和总租赁数量的关系
#绘制第一个子图
plt.figure(1, figsize=(14, 8))
plt.subplot(221)
hour_casual = df[df.holiday==1].groupby('hour')['casual'].median()
hour_registered = df[df.holiday==1].groupby('hour')['registered'].median()
hour_count = df[df.holiday==1].groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.xticks(hour_casual.index)
plt.title('未注册用户和注册用户在节假日自行车租赁情况')
#绘制第二个子图
plt.subplot(222)
hour_casual = df[df.workingday==1].groupby('hour')['casual'].median()
hour_registered = df[df.workingday==1].groupby('hour')['registered'].median()
hour_count = df[df.workingday==1].groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.title('未注册用户和注册用户在工作日自行车租赁情况')
plt.xticks(hour_casual.index)
#绘制第三个子图
plt.subplot(212)
hour_casual = df.groupby('hour')['casual'].median()
hour_registered = df.groupby('hour')['registered'].median()
hour_count = df.groupby('hour')['count'].median()
plt.plot(hour_casual.index, hour_casual.values, '-', color='r', label='未注册用户')
plt.plot(hour_registered.index, hour_registered.values, '-', color='g', label='注册用户')
plt.plot(hour_count.index, hour_count.values, '-o', color='c', label='所有用户')
plt.legend()
plt.title('未注册用户和注册用户自行车租赁情况')
plt.xticks(hour_casual.index)
plt.show()

# 天气和总租赁数量
ig, ax = plt.subplots(3, 1, figsize=(12, 6))
sns.boxplot(x='weather', y='casual', hue='workingday',data=df, ax=ax[0])
sns.boxplot(x='weather', y='registered',hue='workingday', data=df, ax=ax[1])
sns.boxplot(x='weather', y='count',hue='workingday', data=df, ax=ax[2])

df[df.weather==4]
sns.boxplot(x='season', y='month',data=df)

# 季节的划分通常和纬度相关, 而这份数据是用来预测美国华盛顿的租赁数量, 且美国和我国的纬度基本一样, 故按照345春节, 678夏季..这个规则来重新划分
import numpy as np
df['group_season'] = np.where((df.month <=5) & (df.month >=3), 1,
                        np.where((df.month <=8) & (df.month >=6), 2,
                                 np.where((df.month <=11) & (df.month >=9), 3, 4)))
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
#绘制气温和季节箱线图
sns.boxplot(x='season', y='temp',data=df, ax=ax[0])
sns.boxplot(x='group_season', y='temp',data=df, ax=ax[1])

df.drop('season', axis=1, inplace=True)
df.shape

# 其他变量和总租赁数量的关系
sns.pairplot(df[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])

df['windspeed']

df.loc[df.windspeed == 0, 'windspeed'] = np.nan
df.fillna(method='bfill', inplace=True)
df.windspeed.isnull().sum()

# 相关矩阵
#对数转换
df['windspeed'] = np.log(df['windspeed'].apply(lambda x: x+1))
df['casual'] = np.log(df['casual'].apply(lambda x: x+1))
df['registered'] = np.log(df['registered'].apply(lambda x: x+1))
df['count'] = np.log(df['count'].apply(lambda x: x+1))
sns.pairplot(df[['windspeed', 'casual', 'registered', 'count']])

# 经过对数变换之后, 注册用户和所有用户的租赁数量和正态还是相差较大, 故在计算相关系数时选择spearman相关系数
correlation = df.corr(method='spearman')
plt.figure(figsize=(12, 8))
#绘制热力图
sns.heatmap(correlation, linewidths=0.2, vmax=1, vmin=-1, linecolor='w',
            annot=True,annot_kws={'size':8},square=True)

# 岭回归划分数据集
from sklearn.model_selection import train_test_split
#由于所有用户的租赁数量是由未注册用户和注册用户相加而成, 故删除.
df.drop(['casual','registered'], axis=1, inplace=True)
X = df.drop(['count'], axis=1)
y = df['count']
#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# 模型训练
from sklearn.linear_model import Ridge
#这里的alpha指的是正则化项参数, 初始先设置为1.
rd = Ridge(alpha=1)
rd.fit(X_train, y_train)
print(rd.coef_)
print(rd.intercept_)

# 通过岭迹图来选择正则化参数
#设置参数以及训练模型
alphas = 10**np.linspace(-5, 10, 500)
betas = []
for alpha in alphas:
    rd = Ridge(alpha = alpha)
    rd.fit(X_train, y_train)
    betas.append(rd.coef_)
#绘制岭迹图
plt.figure(figsize=(8,6))
plt.plot(alphas, betas)
#对数据进行对数转换, 便于观察.
plt.xscale('log')
#添加网格线
plt.grid(True)
#坐标轴适应数据量
plt.axis('tight')
plt.title(r'正则化项参数$\alpha$和回归系数$\beta$岭迹图')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

# 交叉验证的岭回归
from sklearn.linear_model import RidgeCV
from sklearn import metrics
rd_cv = RidgeCV(alphas=alphas, cv=10, scoring='r2')
rd_cv.fit(X_train, y_train)
rd_cv.alpha_

# 选出的最佳正则化项参数为805.03, 然后用这个参数进行模型训练
rd = Ridge(alpha=805.0291812295973) #, fit_intercept=False
rd.fit(X_train, y_train)
print(rd.coef_)
print(rd.intercept_)

# 模型预测
from sklearn import metrics
from math import sqrt
#分别预测训练数据和测试数据
y_train_pred = rd.predict(X_train)
y_test_pred = rd.predict(X_test)
#分别计算其均方根误差和拟合优度
y_train_rmse = sqrt(metrics.mean_squared_error(y_train, y_train_pred))
y_train_score = rd.score(X_train, y_train)
y_test_rmse = sqrt(metrics.mean_squared_error(y_test, y_test_pred))
y_test_score = rd.score(X_test, y_test)
print('训练集RMSE: {0}, 评分: {1}'.format(y_train_rmse, y_train_score))
print('测试集RMSE: {0}, 评分: {1}'.format(y_test_rmse, y_test_score))

# Lasso回归模型训练
from sklearn.linear_model import Lasso
alphas = 10**np.linspace(-5, 10, 500)
betas = []
for alpha in alphas:
    Las = Lasso(alpha = alpha)
    Las.fit(X_train, y_train)
    betas.append(Las.coef_)
plt.figure(figsize=(8,6))
plt.plot(alphas, betas)
plt.xscale('log')
plt.grid(True)
plt.axis('tight')
plt.title(r'正则化项参数$\alpha$和回归系数$\beta$的Lasso图')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

# 交叉验证选择Lasso回归最优正则化项参数
from sklearn.linear_model import LassoCV
from sklearn import metrics
Las_cv = LassoCV(alphas=alphas, cv=10)
Las_cv.fit(X_train, y_train)
Las_cv.alpha_

# 用这个参数重新训练模型
Las = Lasso(alpha=0.005074705239490466) #, fit_intercept=False
Las.fit(X_train, y_train)
print(Las.coef_)
print(Las.intercept_)

# 对比岭回归可以发现, 这里的回归系数中有0存在, 也就是舍弃了holiday, workingday, weather和group_season这四个自变量
#用Lasso分别预测训练集和测试集, 并计算均方根误差和拟合优度
y_train_pred = Las.predict(X_train)
y_test_pred = Las.predict(X_test)
y_train_rmse = sqrt(metrics.mean_squared_error(y_train, y_train_pred))
y_train_score = Las.score(X_train, y_train)
y_test_rmse = sqrt(metrics.mean_squared_error(y_test, y_test_pred))
y_test_score = Las.score(X_test, y_test)
print('训练集RMSE: {0}, 评分: {1}'.format(y_train_rmse, y_train_score))
print('测试集RMSE: {0}, 评分: {1}'.format(y_test_rmse, y_test_score))

# 用传统的线性回归进行预测, 从而对比三者之间的差异
from sklearn.linear_model import LinearRegression
#训练线性回归模型
LR = LinearRegression()
LR.fit(X_train, y_train)
print(LR.coef_)
print(LR.intercept_)
#分别预测训练集和测试集, 并计算均方根误差和拟合优度
y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)
y_train_rmse = sqrt(metrics.mean_squared_error(y_train, y_train_pred))
y_train_score = LR.score(X_train, y_train)
y_test_rmse = sqrt(metrics.mean_squared_error(y_test, y_test_pred))
y_test_score = LR.score(X_test, y_test)
print('训练集RMSE: {0}, 评分: {1}'.format(y_train_rmse, y_train_score))
print('测试集RMSE: {0}, 评分: {1}'.format(y_test_rmse, y_test_score))