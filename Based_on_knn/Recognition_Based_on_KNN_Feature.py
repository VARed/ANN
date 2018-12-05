# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:34:02 2018

@author: Administrator
"""
import numpy
from sklearn.neighbors.classification import KNeighborsClassifier
from Dataset import Dataset

def main(filedir,k,setting=0):
    error=1
    corr=0
    if setting:
        Train=Dataset(filedir, 'train', 1500,feature_set=1,feature_num=5,code_num=5,maxy=24,miny=5,maxx=17,minx=5,distance=24)
        Train.data()
        Test= Dataset(filedir, 'test', 500,feature_set=1,feature_num=5,code_num=5,maxy=24,miny=5,maxx=17,minx=5,distance=24)
        Test.data()
    train_set=numpy.load(filedir+'train_set.npy')
    train_label=numpy.load(filedir+'train_label.npy')
    test_set=numpy.load(filedir+'train_set.npy')
    test_label=numpy.load(filedir+'train_label.npy')
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
main(k=13,setting=0,filedir='D:/ANN/DataSet/')