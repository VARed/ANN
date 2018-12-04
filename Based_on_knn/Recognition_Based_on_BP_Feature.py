# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:44:52 2018

@author: Administrator
"""

from random import randrange
from random import random
from math import exp
import numpy


class BP_Network():
    # 初始化神经网络
    def __init__(self, n_inputs, n_hidden, n_outputs, setting):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        if setting:
            self.network = list()
            hidden_layer = [{'weights': [random() for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden)]
            self.network.append(hidden_layer)
            output_layer = [{'weights': [random() for i in range(self.n_hidden + 1)]} for i in range(self.n_outputs)]
            self.network.append(output_layer)
        else:
            self.network = numpy.load("D:/ANN/DataSet/weight_set.npy")

    # 计算神经元的激活值（加权之和）
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    # 定义激活函数
    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # 计算神经网络的正向传播
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # 计算激活函数的导数
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # 反向传播误差信息，并将纠偏责任存储在神经元中
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['responsibility'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['responsibility'] = errors[j] * self.transfer_derivative(neuron['output'])

    # 根据误差，更新网络权重
    def _update_weights(self, train):
        for i in range(len(self.network)):
            inputs = train
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.l_rate * neuron['responsibility'] * inputs[j]
                neuron['weights'][-1] += self.l_rate * neuron['responsibility']

    # 根据指定的训练周期训练网络
    def train_network(self, train, label):
        for epoch in range(self.n_epoch):
            sum_error = 0
            for k in range(len(train_set)):
                outputs = self.forward_propagate(train_set[k])
                expected = [0 for i in range(self.n_outputs)]
                expected[label[k]] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self._update_weights(train[k])
            print('>周期=%d, 误差=%.3f' % (epoch, sum_error))

    # 利用训练好的网络，预测“新”数据
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))

    # 利用随机梯度递减策略，训练网络
    def back_propagation(self, train_set, train_label, test_set):
        self.train_network(train_set, train_label)
        predictions = list()
        for k in range(len(test_set)):
            prediction = self.predict(test_set[k])
            predictions.append(prediction)
        return (predictions)

    # 用预测正确百分比来衡量正确率
    def accuracy_metric(self, actual, predicted):
        correct = 0
        error = 1
        for i in range(len(actual) // 5):
            for k in range(5):
                if actual[i * 5 + k] != predicted[i * 5 + k]:
                    error = 0
            if error:
                correct = correct + 1
            error = 1
        return correct / float(len(actual) // 5) * 100.0

    # 用每一个交叉分割的块（训练集合，试集合）来评估BP算法
    def evaluate_algorithm(self, train_set, train_label, test_set, test_label, l_rate, n_epoch):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.train_set = train_set
        self.train_label = train_label
        self.test_set = test_set
        self.test_label = test_label
        predicted = self.back_propagation(train_set, train_label, test_set)
        accuracy = self.accuracy_metric(test_label, predicted)
        print('整个验证码识别准确率为:', accuracy)


def normalize_dataset(dataset):
    minmax = [[min(column), max(column)] for column in zip(*dataset)]
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


if __name__ == '__main__':
    # 设置随机种子
    # 构建训练数据    
    train_set = numpy.load("D:/ANN/DataSet/train_set.npy")
    train_label = numpy.load("D:/ANN/DataSet/train_label.npy")
    test_set = numpy.load("D:/ANN/DataSet/test_set.npy")
    test_label = numpy.load("D:/ANN/DataSet/test_label.npy")
    normalize_dataset(train_set)
    normalize_dataset(test_set)
    # 设置网络初始化参数
    n_inputs = len(train_set[0])
    n_hidden = 10
    n_outputs = 10
    BP = BP_Network(n_inputs, n_hidden, n_outputs, 0)
    l_rate = 0.2
    n_epoch = 0
    BP.evaluate_algorithm(train_set, train_label, test_set, test_label, l_rate, n_epoch)
    numpy.save("D:/ANN/DataSet/weight_set.npy", BP.network)


