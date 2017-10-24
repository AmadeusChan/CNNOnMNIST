import numpy as np
import functions

input = np.arange(150).reshape(2,3,5,5)
W = np.arange(60).reshape(5,3,2,2)
b = np.arange(5)
k = 2
pad = 2

a = np.arange(32).reshape(2,2,2,4)
print a
print functions.avgpool2d_forward(a, 2, 2)

'''
print functions.conv2d_forward(input, W, b, k, pad)
print '------'
print functions.conv_forward(input, W, b, k, pad)
'''

'''
output = np.arange(640).reshape(2, 5, 8, 8)
print functions.conv2d_backward(input, output, W, b, k, pad)
'''
