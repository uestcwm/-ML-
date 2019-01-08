import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('ex1data2.txt',names=['population','numb','profit'])#对应数据中大小房间数收益

"""特征归一化"""
data2=(data-data.mean())/data.std()#
print(data2)

"""代价函数"""
def computecost(x,y,theta):#以theta为变量
    inner=np.power(((x*theta.T)-y),2)#平方项,这里要转置不然维度不匹配
    return np.sum(inner)/(2*len(x))#返回J(theta)  =(1/2m)sum((x.*theta.T)-y)^2

"""梯度下降法"""
def gradientdescent(x,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))#产生一个跟theta一样大的zeros矩阵
    parameters=int(theta.ravel().shape[1])#shape[1]第二维的长度,ravel将多维数组变成一维然后求列数
    '''eg:theta=[[1,2,3,4],[5,6,7,8]]变成[1，2，3，4，5，6，7，8]'''
    cost=np.zeros(iters)#每一次的迭代放入一个cost矩阵

    for i in range(iters):
        error=(x*theta.T)-y#这句话只是每次计算一遍

        for j in range(parameters):
            term=np.multiply(error,x[:,j])#multiply对应位置元素相乘
            temp[0,j]=theta[0,j]-((alpha/len(x))*np.sum(term))

            theta=temp
            cost[i]=computecost(x,y,theta)
    return theta,cost

"""数据预处理：增加一列/分离自变量因变量/转换为矩阵"""
data2.insert(0,'ones',1)#在0和1之间增加一列

cols=data2.shape[1]#第一维度
x=data2.iloc[:,0:cols-1]
y=data2.iloc[:,cols-1:cols]
x=np.matrix(x.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0,0]))
alpha=0.01
iters=1000

g2,cost2=gradientdescent(x,y,theta,alpha,iters)

computecost(x,y,g2)

'''绘制代价函数曲线'''
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost2,'r')
ax.set_xlabel('iterations')
ax.set_ylabel('cost')
ax.set_title('error')
plt.show()

'''sklearn包自带线性回归函数LinearRegression预测'''
from sklearn import linear_model
model=linear_model.LinearRagression()
model.fit(x,y)

x=np.array(x[:,1].A1)#A1
f=model.predict(x).flatten()

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(x,f,'b',label='prediction')
ax.scatter(data.population,data.profit,label='traning data')
ax.legend(loc=2)#图例的位置2：upper right
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.set_title('predicted vs profit')
plt.show()










