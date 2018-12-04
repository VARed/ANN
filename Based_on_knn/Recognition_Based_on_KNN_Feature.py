# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:34:02 2018

@author: Administrator
"""
import numpy
from sklearn.neighbors.classification import KNeighborsClassifier
from Dataset import Dataset

def main(k,setting=0):
    error=1
    corr=0
    if setting:
        Train=Dataset('D:/ANN/DataSet/','train',1500)
        Train.data()
        Test= Dataset('D:/ANN/DataSet/', 'test', 500)
        Test.data()
    train_set=numpy.load("D:/ANN/DataSet/train_set.npy")
    train_label=numpy.load("D:/ANN/DataSet/train_label.npy")
    test_set=numpy.load("D:/ANN/DataSet/test_set.npy")
    test_label=numpy.load("D:/ANN/DataSet/test_label.npy")
    KNN=KNeighborsClassifier(n_neighbors=k)
    KNN.fit(train_set,train_label)
    result=KNN.predict(test_set)
    for k in range(len(result)//5):
        for i in range(5):
            if result[k*5+i]!=test_label[k*5+i]:
                error=0
        if error:
            corr=corr+1
        error=1
    print('整个验证码识别准确率为:', corr/(len(result)//5)*100.0)
main(13,0)