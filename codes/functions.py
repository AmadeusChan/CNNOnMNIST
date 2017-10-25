import numpy as np
from scipy import signal

import skimage.measure
import scipy.ndimage
from skimage.util import view_as_windows as viewW
import utils
import im2colfun

import numpy as np

try:
  from im2col_cython import col2im_cython, im2col_cython
  from im2col_cython import col2im_6d_cython
except ImportError:
  print 'run the following from the cs231n directory and try again:'
  print 'python setup.py build_ext --inplace'
  print 'You may also need to restart your iPython kernel'

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1
  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
  return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=0, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols

def im2col_sliding_strided(A, BSZ):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)

def im2col_each(x,hh,ww,stride=1):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def im2col(input, k_x, k_y):
    c_in, h_in, w_in = input.shape
    h_out = h_in - k_x + 1
    w_out = w_in - k_y + 1
 
    return im2col_indices(input.reshape(1, c_in, h_in, w_in), k_x, k_y)

    # return im2col_each(input, k_x, k_y).transpose(1, 0)

    result = np.ndarray(shape = (0, h_out * w_out))
    for c in range(c_in):
    	# temp = viewW(input[c], (k_x, k_y)).reshape(-1, k_x * k_y).T
        temp = im2col_sliding_strided(input[c], [k_x, k_y])
        # temp = im2col_each(input[c], k_x, k_y)
        result = np.append(result, temp, axis = 0)
    return result

def conv(input, W):
    
    # utils.check.goin()

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
    
    # image = im2col_indices(input, k_x, k_y)
    image = im2col_cython(input, k_x, k_y, 0, 1)
    output = np.dot(W, image)
    return output.reshape(c_out, N, h_out, w_out).reshape(c_out, h_out, w_out, N).transpose(3, 0, 1, 2)

    '''
    image = input.transpose(1, 0, 2, 3)
    image = image.reshape(c_in, N * h_in, w_in)

    image = np.lib.pad(image, ((0, 0), (0, k_x - 1), (0, 0)), 'constant')

    # utils.check.out_conv_for();
    image = im2col(image, k_x, k_y)
    output = np.dot(W, image)
    output = output.reshape(c_out, N * h_in, w_out)
    output = output.reshape(c_out, N, h_in, w_out)
    if k_x > 1:
        output = output[:, :, :-(k_x - 1) , :]
    # print output.shape
    return output.transpose(1, 0, 2, 3)
    '''


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

