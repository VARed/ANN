# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:34:02 2018

@author: Administrator
"""
import datetime
import numpy
from sklearn.neighbors.classification import KNeighborsClassifier
from Dataset import Dataset
from sklearn.model_selection import train_test_split
def main(filedir,k,setting=0):
    starttime = datetime.datetime.now()
    print('Start:\t\t' + starttime.strftime("%Y-%m-%d %X"))
    error=1
    corr=0
    if setting:
        Data_set = Dataset(filedir, 2000,feature_set=1,feature_num=5,code_num=5,maxy=24,miny=5,maxx=17,minx=5,distance=24)
        Data_set.data()
    dataset = numpy.load(filedir+'set.npy')
    label = numpy.load(filedir+'label.npy')
    train_set, test_set, train_label, test_label = train_test_split(dataset, label, test_size=0.25)
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
    endtime = datetime.datetime.now()
    print('End:\t\t' + endtime.strftime("%Y-%m-%d %X"))
    duringtime = endtime - starttime
    print('Spend Time:\t' + str(duringtime))
main(k=13,setting=0,filedir='./DataSet/')