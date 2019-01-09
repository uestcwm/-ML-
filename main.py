import pandas as pd
import numpy as np
import seaborn as sns#可以看做matplotlib一样的功能
sns.set(context="notebook",style="whitegrid",palette="dark")
import matplotlib.pyplot as plt

data=pd.read_csv('ex1data1.txt',names=['population','profit'])#读取数据并赋给每列名字
#print(df.describe())#一些属性的显示
#print(df.info())#info：查看属性索引数据类型大小
#print(df.head())#head:查看前五行

"""观察原始数据，知道大致分布情况"""
sns.lmplot('population','profit',data,size=6,fit_reg=False)#参数：x,y点的大小
plt.show()

"""代价函数"""
def computecost(x,y,theta):#以theta为变量
    inner=np.power(((x*theta.T)-y),2)#平方项,这里要转置不然维度不匹配
    return np.sum(inner)/(2*len(x))#返回J(theta)  =(1/2m)sum((x.*theta.T)-y)^2

"""在两列的训练集中增加一列，使用向量化的方法来计算代价和梯度"""
data.insert(0,'ones',1)#在第一列和第二列中间插入ones

"""变量初始化x:training data y:target variable"""
col=data.shape[1]#1：第二维度即列数这里是3
print(col)
x=data.iloc[:,0:col-1]#去掉最后一列的所有行#iloc选取列从0开始计数
y=data.iloc[:,col-1:col]#最后一列的所有行target variable
print(x.head())#这两句检验一下是否正确
print(y.head())#head()输出前五个

"""现在的数据类型是DataFrame转换为matrix"""
x=np.matrix(x.values)#转换
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))#theta是一个二维变量2个参数
print(computecost(x,y,theta))#计算初始值看看

"""梯度下降法"""
def gradientdescent(x,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))#产生给定形状和类型的用0填充的跟theta一样大的数组转换为矩阵
    parameters=int(theta.ravel().shape[1])#shape[1]第二维的长度,ravel将多维数组变成一维然后求列数
    '''eg:theta=[[1,2,3,4],[5,6,7,8]]变成[1，2，3，4，5，6，7，8]'''
    cost=np.zeros(iters)#每一次的迭代放入一个cost矩阵

    for i in range(iters):
        error=(x*theta.T)-y#这句话只是每次计算一遍

        for j in range(parameters):
            term=np.multiply(error,x[:,j])#multiply对应位置元素相乘
            temp[0,j]=theta[0,j]-((alpha/len(x))*np.sum(term))#求过导之后的式子

            theta=temp
            cost[i]=computecost(x,y,theta)
    return theta,cost

"""开始单变量线性回归：初始化变量"""
alpha=0.01
iters=1000
g,cost=gradientdescent(x,y,theta,alpha,iters)#g是theta,对应梯度下降返回值

computecost(x,y,g)

"""根据拟合好的g即theta绘制曲线"""
x=np.linspace(data.population.min(),data.population.max(),100)#linspace用于创建等差数列100指元素个数
f=g[0,0]+(g[0,1]*x)
fig,ax=plt.subplots(figsize=(12,8))#figsize定义图像窗口的大小
ax.plot(x,f,'r',label='prediction')
ax.scatter(data.population,data.profit,label='traning data')
ax.legend(loc=2)#位置2：upper right corner
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.set_title('prediction')
plt.show()

"""绘制代价函数"""
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('iterations')
ax.set_ylabel('cost')
ax.set_title('training erroe')
plt.show()