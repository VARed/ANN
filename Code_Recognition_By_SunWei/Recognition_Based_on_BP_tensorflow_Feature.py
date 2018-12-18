# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 18:50:39 2018

@author: SunWei
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from Dataset import Dataset
def main(filedir,setting=0):
    num_classes = 10  # 输出大小
    input_size = 5  # 输入大小
    hidden_units_size = 8  # 隐藏层节点数量
    training_iterations = 1000
    X = tf.placeholder(tf.float32, shape = [None, input_size])
    Y = tf.placeholder(tf.float32, shape = [None, num_classes])
    W1 = tf.Variable(tf.random_normal ([input_size, hidden_units_size], stddev = 0.1))
    B1 = tf.Variable(tf.constant (0.1), [hidden_units_size])
    W2 = tf.Variable(tf.random_normal ([hidden_units_size, num_classes], stddev = 0.1))
    B2 = tf.Variable(tf.constant (0.1), [num_classes])
    hidden_opt = tf.matmul(X, W1) + B1  # 输入层到隐藏层正向传播
    hidden_opt = tf.nn.relu(hidden_opt)  # 激活函数，用于计算节点输出值
    final_opt = tf.matmul(hidden_opt, W2) + B2  # 隐藏层到输出层正向传播
    # 对输出层计算交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_opt))
    # 梯度下降算法，这里使用了反向传播算法用于修改权重，减小损失
    opt = tf.train.GradientDescentOptimizer(1).minimize(loss)
    # 初始化变量
    init = tf.global_variables_initializer()
    # 计算准确率
    correct_prediction =tf.equal (tf.argmax (Y, 1), tf.argmax(final_opt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    sess = tf.Session ()
    sess.run (init)
    for k in range (training_iterations) :
        Data_set = Dataset(filedir, 2000, feature_set=1, feature_num=5)
        if setting:
            Data_set.data()
        dataset= np.load(filedir+'set.npy')
        Originallabel= np.load(filedir+'label.npy')
        train_set, test_set, train_label, test_label = train_test_split(dataset, Originallabel, test_size=0.25)
        testlabel=np.zeros(shape=(len(test_label),10))
        trainlabel=np.zeros(shape=(len(train_label),10))
        for i in range(len(test_label)):
            testlabel[i]=[float(test_label[i]==j) for j in range(10)]
        for i in range(len(train_label)):
            trainlabel[i]=[float(train_label[i]==j) for j in range(10)]
        # 训练
        sess.run ([opt, loss], feed_dict = {X: train_set, Y: trainlabel})
        train_accuracy = accuracy.eval (session = sess, feed_dict = {X: test_set, Y: testlabel})
        print ("step : %d, training accuracy = %g " % (k, train_accuracy))
main(setting=0,filedir='./DataSet2/')