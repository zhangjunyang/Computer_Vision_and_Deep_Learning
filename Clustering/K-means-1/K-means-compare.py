
#!/usr/bin/env python
# -*- coding:utf-8 -*-
 
#kmean与mini batch kmeans 算法的比较
 
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin
 
#解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
 
#初始化三个中心
centers = [[1,1],[-1,-1],[1,-1]]
clusters = len(centers)     #聚类数目为3
#产生3000组二维数据样本，三个中心点，标准差是0.7
X,Y = make_blobs(n_samples=300,centers=centers,cluster_std=0.7,random_state=28)
 
#构建kmeans算法
k_means =  KMeans(init="k-means++",n_clusters=clusters,random_state=28)
t0 = time.time()
k_means.fit(X)      #模型训练
km_batch = time.time()-t0       #使用kmeans训练数据消耗的时间
print("K-Means算法模型训练消耗时间:%.4fs"%km_batch)
 
#构建mini batch kmeans算法
batch_size = 100        #采样集的大小
mbk = MiniBatchKMeans(init="k-means++",n_clusters=clusters,batch_size=batch_size,random_state=28)
t0 = time.time()
mbk.fit(X)
mbk_batch = time.time()-t0
print("Mini Batch K-Means算法模型训练消耗时间:%.4fs"%mbk_batch)
 
#预测结果
km_y_hat = k_means.predict(X)
mbk_y_hat = mbk.predict(X)
 
#获取聚类中心点并对其排序
k_means_cluster_center = k_means.cluster_centers_
mbk_cluster_center = mbk.cluster_centers_
print("K-Means算法聚类中心点:\n center=",k_means_cluster_center)
print("Mini Batch K-Means算法聚类中心点:\n center=",mbk_cluster_center)
order = pairwise_distances_argmin(k_means_cluster_center,mbk_cluster_center)
 
#画图
plt.figure(figsize=(12,6),facecolor="w")
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9)
cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
 
#子图1——原始数据
plt.subplot(221)
plt.scatter(X[:,0],X[:,1],c=Y,s=6,cmap=cm,edgecolors="none")
plt.title(u"原始数据分布图")
plt.xticks(())
plt.yticks(())
plt.grid(True)
 
#子图2：K-Means算法聚类结果图
plt.subplot(222)
plt.scatter(X[:,0], X[:,1], c=km_y_hat, s=6, cmap=cm,edgecolors='none')
plt.scatter(k_means_cluster_center[:,0], k_means_cluster_center[:,1],c=range(clusters),s=60,cmap=cm2,edgecolors='none')
plt.title(u'K-Means算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-3.8, 3,  'train time: %.2fms' % (km_batch*1000))
plt.grid(True)
 
#子图三Mini Batch K-Means算法聚类结果图
plt.subplot(223)
plt.scatter(X[:,0], X[:,1], c=mbk_y_hat, s=6, cmap=cm,edgecolors='none')
plt.scatter(mbk_cluster_center[:,0], mbk_cluster_center[:,1],c=range(clusters),s=60,cmap=cm2,edgecolors='none')
plt.title(u'Mini Batch K-Means算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-3.8, 3,  'train time: %.2fms' % (mbk_batch*1000))
plt.grid(True)
plt.savefig("kmean与mini batch kmeans 算法的比较.png")
plt.show()
