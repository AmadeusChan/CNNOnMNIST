import numpy as np
from scipy import signal

import skimage.measure
import scipy.ndimage
from skimage.util import view_as_windows as viewW
import utils
import im2colfun

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

def im2col(input, k_x, k_y):
    c_in, h_in, w_in = input.shape
    h_out = h_in - k_x + 1
    w_out = w_in - k_y + 1
    result = np.ndarray(shape = (0, h_out * w_out))
    for c in range(c_in):
    	# temp = viewW(input[c], (k_x, k_y)).reshape(-1, k_x * k_y).T
        temp = im2col_sliding_strided(input[c], [k_x, k_y])
        result = np.append(result, temp, axis = 0)
    return result

def conv(input, W):
    
    c_out = W.shape[0]
    c_in = W.shape[1]
    k_x = W.shape[2]
    k_y = W.shape[3]
    N = input.shape[0]
    h_in = input.shape[2]
    w_in = input.shape[3]
    h_out = h_in - k_x + 1
    w_out = w_in - k_y + 1

    output = np.zeros(shape = (N, c_out, h_out, w_out))

    W = W.reshape(c_out, c_in * k_x * k_y) # of shape c_out x (c_in x k x k)

    for n in range(N):
        image = input[n] 
        image = im2col(image, k_x, k_y) # now of shape (c_in x k x k) x (h_out x w_out)
        fm = np.dot(W, image) # of shape c_out x (h_out x w_out)
        output[n] = fm.reshape(c_out, h_out, w_out)
        for c in range(c_out):
            output[n][c] = output[n][c] 

    return output

def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution

    note that:
        h_out = h_in + 2 x pad - kernel + 1
        w_out = w_in + 2 x pad - kernel + 1
    '''
    utils.check.goin()

    input = np.lib.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    # print input

    N, c_in, h_in, w_in = input.shape
    c_out = W.shape[0]
    k = W.shape[2]

    # print input.shape 

    h_out = h_in - k + 1
    w_out = w_in - k + 1

    # print h_out, ' ', w_out,' ',h_in,' ',w_in
    
    # output = np.zeros(shape = (N, c_out, h_out, w_out))

    # print output.shape

    '''
    naive implement
    '''
    '''
    for n in range(N):
        for co in range(c_out):
            output[n][co] = output[n][co] + b[co]
            for ci in range(c_in):
                image = input[n][ci]
                f = np.rot90(W[co][ci], 2)
                feature_map = signal.convolve2d(image, f, mode = 'valid')
                output[n][co] = output[n][co] + feature_map

    utils.check.out_conv_for()
    return output
    '''
    '''
    faster implement with im2col
    '''

    '''
    W = W.reshape(c_out, c_in * k * k) # of shape c_out x (c_in x k x k)

    for n in range(N):
        image = input[n] 
        image = im2col(image, k) # now of shape (c_in x k x k) x (h_out x w_out)
        fm = np.dot(W, image) # of shape c_out x (h_out x w_out)
        output[n] = fm.reshape(c_out, h_out, w_out)
        for c in range(c_out):
            output[n][c] = output[n][c] 
    '''

    output = conv(input, W)
    output = output + b.reshape(1, c_out, 1, 1).repeat(h_out, axis = 2).repeat(w_out, axis = 3).repeat(N, axis = 0)
    utils.check.out_conv_for()
    return output

    # pass


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''

    '''
    naive implement
    '''

    utils.check.goin()
    N, c_in, h_in, w_in = input.shape
    c_out = W.shape[0]
    k = W.shape[2]

    input = np.lib.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    grad_input = np.zeros(shape = (N, c_in, h_in, w_in))
    grad_W = np.zeros(shape = (c_out, c_in, k, k))
    grad_b = np.zeros(c_out)

    grad_out = grad_output

    kernel = kernel_size
    grad_out = np.lib.pad(grad_out, ((0, 0), (0, 0), (kernel - 1, kernel - 1), (kernel - 1, kernel - 1)), 'constant')
    W = np.rot90(W, 2, axes = (2, 3))
    W = W.transpose(1, 0, 2, 3)

    grad_input = conv(grad_out, W)
    if pad > 0:
        grad_input = grad_input[:, :, pad: -pad, pad: -pad]
    
    grad_W = conv(input.transpose(1, 0, 2, 3), grad_output.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)
    grad_b = np.sum(grad_output, axis = (0, 2, 3))

    utils.check.out_conv_bac()
   
    return grad_input, grad_W, grad_b
    # pass


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input

    note that:
        h_out = (h_in + pad * 2) / kernel_size
        w_out = (w_in + pad * 2) / kernel_size
    '''
    utils.check.goin()
    input = np.lib.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    k = kernel_size
    output = skimage.measure.block_reduce(input, (1, 1, k, k), np.mean)
    # print output.shape
    utils.check.out_pool_for()
    return output
    # pass


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    utils.check.goin()
    # output = np.zeros(shape = input.shape)
    # N = input.shape[0]
    # c_in = input.shape[1]
    k = kernel_size
    output = grad_output.repeat(k, axis = 2).repeat(k, axis = 3)
    if pad > 0:
    	output = output[:, :, pad: -pad, pad: -pad]
    output = output / (k * k)
    '''
    for n in range(N):
        for c in range(c_in):
            temp = scipy.ndimage.zoom(grad_output[n][c], kernel_size, order=0) / (kernel_size * kernel_size)
            if pad > 0:
                temp = temp[pad: -pad, pad: -pad]
            output[n][c] = temp
    '''

    utils.check.out_pool_bac()
    return output

    # pass

