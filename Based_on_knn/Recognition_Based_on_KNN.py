# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:34:02 2018

@author: Administrator
"""
import numpy
import cv2
from fnmatch import fnmatch
import os
from sklearn.neighbors.classification import KNeighborsClassifier
#图片预处理
#提取图片特征
def feature(A):
    midx=int(A.shape[1]/2)+1
    midy=int(A.shape[0]/2)+1
    A1=A[0:midy,0:midx].mean()
    A2=A[midy:A.shape[0],0:midx].mean()
    A3=A[0:midy,midx:A.shape[1]].mean()
    A4=A[midy:A.shape[0],midx:A.shape[1]].mean()
    #A5=A.mean()
    A5=A[midy-1:midy+2,midx-1:midx+2].mean()
    AF=[A1,A2,A3,A4,A5]
    return AF
#切割图片并返回每个子图片特征
def incise(im):
    AF=[]
    for i in range(5):
        AF.append(feature(im[5:24,i*24+5:i*24+17]))
    return AF
#训练已知图片的特征         
def training():
    train_set=numpy.zeros(shape=(1500*5,5))
    k=0
    label=[]
    filedir = 'D:/ANN/DataSet/trainset'
    for file in os.listdir(filedir):
        if fnmatch(file, '*.jpg'):
            img_name = file
            im,imagedata= _get_dynamic_binary_image(filedir, img_name)
            im = interference_line(im)
            im = interference_point(im)
            for i in range(5):
                train_set[k*5+i]=feature(im[5:24,i*24+5:i*24+17])
                label.append(int(img_name.split('.')[0][i]))
            k=k+1
    numpy.save("D:/ANN/DataSet/train_label.npy",label)
    numpy.save("D:/ANN/DataSet/train_set.npy",train_set)
    return train_set,label

def clear_border(img):
  '''去除边框'''
  h, w = img.shape[:2]
  for y in range(0, w):
    for x in range(0, h):
      # if y ==0 or y == w -1 or y == w - 2:
      if y < 4 or y > w -4:
        img[x, y] = 255
      # if x == 0 or x == h - 1 or x == h - 2:
      if x < 4 or x > h - 4:
        img[x, y] = 255
  return img
def interference_line(img):
  '''干扰线降噪'''
  h, w = img.shape[:2]
  for y in range(1, w - 1):
    for x in range(1, h - 1):
      count = 0
      if img[x, y - 1] > 245:
        count = count + 1
      if img[x, y + 1] > 245:
        count = count + 1
      if img[x - 1, y] > 245:
        count = count + 1
      if img[x + 1, y] > 245:
        count = count + 1
      if count > 2:
        img[x, y] = 255
  return img
def interference_point(img, x = 0, y = 0):
    """点降噪 9邻域框,以当前点为中心的田字框,黑点个数"""
    # todo 判断图片的长宽度下限
    cur_pixel = img[x,y]# 当前像素点的值
    height,width = img.shape[:2]
    for y in range(0, width - 1):
      for x in range(0, height - 1):
        if y == 0:  # 第一行
            if x == 0:  # 左上顶点,4邻域
                # 中心点旁边3个点
                sum = int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右上顶点
                sum = int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            else:  # 最上非顶点,6邻域
                sum = int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 3 * 245:
                  img[x, y] = 0
        elif y == width - 1:  # 最下面一行
            if x == 0:  # 左下顶点
                # 中心点旁边3个点
                sum = int(cur_pixel) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x, y - 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右下顶点
                sum = int(cur_pixel) \
                      + int(img[x, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y - 1])

                if sum <= 2 * 245:
                  img[x, y] = 0
            else:  # 最下非顶点,6邻域
                sum = int(cur_pixel) \
                      + int(img[x - 1, y]) \
                      + int(img[x + 1, y]) \
                      + int(img[x, y - 1]) \
                      + int(img[x - 1, y - 1]) \
                      + int(img[x + 1, y - 1])
                if sum <= 3 * 245:
                  img[x, y] = 0
        else:  # y不在边界
            if x == 0:  # 左边非顶点
                sum = int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])

                if sum <= 3 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右边非顶点
                sum = int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x - 1, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1])

                if sum <= 3 * 245:
                  img[x, y] = 0
            else:  # 具备9领域条件的
                sum = int(img[x - 1, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1]) \
                      + int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 4 * 245:
                  img[x, y] = 0
    return img
def _get_dynamic_binary_image(filedir, img_name):
  '''自适应阀值二值化'''
  img_name = filedir + '/' + img_name
  #print('.....' + img_name)
  im = cv2.imread(img_name)
  im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
  th1 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
  return th1,im
def main(k,setting=0):
    res=0
    corrnum=0
    allnum=0
    if setting:
        train_set,label=training()
    else:
        train_set=numpy.load("D:/ANN/DataSet/train_set.npy")
        label=numpy.load("D:/ANN/DataSet/train_label.npy")
    KNN=KNeighborsClassifier(n_neighbors=k)
    KNN.fit(train_set, label)
    filedir = 'D:/ANN/DataSet/testset'
    for file in os.listdir(filedir):
        if fnmatch(file, '*.jpg'):
            img_name = file
            im,imagedata= _get_dynamic_binary_image(filedir, img_name)
            im = interference_line(im)
            im = interference_point(im)
            AF=incise(im)
            result=KNN.predict(AF)
            for i in range(5):
                res=result[i]*(10**(4-i))+res
            #print('Predict：',res,'Label：',img_name.split('.')[0])
            if res==int(img_name.split('.')[0]):
                corrnum=corrnum+1
            else:
                print(img_name.split('.')[0],res)
            allnum=allnum+1
            res=0
    print(corrnum/allnum*100)
main(13,1)