import numpy as np
from scipy import signal

import skimage.measure
import scipy.ndimage

import utils

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

    N, c_in, h_in, w_in = input.shape
    c_out = W.shape[0]
    k = W.shape[2]

    # print input.shape 

    h_out = h_in - k + 1
    w_out = w_in - k + 1

    # print h_out, ' ', w_out,' ',h_in,' ',w_in
    
    output = np.zeros(shape = (N, c_out, h_out, w_out))

    # print output.shape

    '''
    naive implement
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

    for n in range(N):
        for ci in range(c_in):
            image = input[n][ci]
            for co in range(c_out):
                # to calculate grad_input
                fm = grad_output[n][co]
                f = W[co][ci]
                temp = signal.convolve2d(fm, f, mode = 'full')

                '''
                print fm.shape
                print f.shape
                print temp.shape, ' ', pad
                '''
                if (pad > 0):
                     temp = temp[pad:-pad, pad:-pad]
                '''
                print temp.shape
                '''

                grad_input[n][ci] = grad_input[n][ci] + temp

                # to calculate the gradient of W
                temp = signal.convolve2d(image, np.rot90(fm, 2), mode = 'valid')
                grad_W[co][ci] = grad_W[co][ci] + temp

                # to calculate the gradient of b
                grad_b[co] = grad_b[co] + np.sum(fm)

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

