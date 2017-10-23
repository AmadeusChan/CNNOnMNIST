from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        N, M = input.shape
        temp = np.exp(input)
        error = np.ndarray(shape = [N])
        for n in range(N):
            sum = np.sum(temp[n])
            error[n] = - np.dot(target[n], input[n] - np.log(sum))
        return np.sum(error) / N 
        # pass

    def backward(self, input, target):
        '''Your codes here'''
        N = input.shape[0]
        temp = np.exp(input)
        sum = np.sum(temp, axis = 1)
        inv_sum = 1.0 / sum.reshape(N, 1)
        predict = temp * inv_sum
        return (predict - target) / N
        # pass
