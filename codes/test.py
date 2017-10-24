import numpy as np
import functions

n = 1
c_in = 1
c_out = 4
h_in = 3
w_in = 3
pad = 2
k = 5

input = np.arange(n * c_in * h_in * w_in).reshape(n, c_in, h_in, w_in)
W = np.arange(c_out * c_in * k * k).reshape(c_out, c_in, k, k)
b = np.zeros(shape=(c_out))

print input
print W
print b

output = functions.conv2d_forward(input, W, b, k, pad)
print output

print functions.conv2d_backward(input, np.log(output), W, b, k, pad)

