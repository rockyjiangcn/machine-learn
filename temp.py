#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 20:52:38 2021

@author: rockyjiang
"""

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split#用于划分训练集和测试集
from sklearn.neural_network import MLPClassifier # 多层感知机模型


digits = datasets.load_digits()# 得到手写数字的数据集
print(digits.data.shape)#样本数和特征数
print(digits.target.shape)#标签数
print(digits.images.shape)#图片8*8

def draw():
    #显示前面36个
    for i in range(36):
        plt.subplot(6,6,i+1)#以6行6列进行显示，并从1开始（i=0）
        plt.imshow(digits.images[i])#图片绘制
    mlt.show()#图片显示
    pass
draw()

x = digits.data# 获取特征数据
y = digits.target# 获取标签数据
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)# 20%作为测试集

# 构建mlp模型
mlp = MLPClassifier(hidden_layer_sizes=(300,),
                   activation='relu')
mlp.fit(x_train,y_train)#训练模型
y_predict = mlp.predict(x_test)#得到预测结果
print(y_predict)#预测结果
print(y_test)#真实结果
score = mlp.score(x_test,y_test)
print(score)
