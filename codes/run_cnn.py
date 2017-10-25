from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
import numpy as np
import time
import os
import json
import sys
from scipy import misc

train_data, test_data, train_label, test_label = load_mnist_4d('data')
# train_data = train_data + np.random.randn(*train_data.shape) * 0.01

# Your model defintion here
# You should explore different model architecture
model = Network()

conv1 = Conv2D('conv1', 1, 3, 5, 0, 1) # output shape: N x 3 x 24 x 24
model.add(conv1)
relu1 = Relu('relu1')
model.add(relu1)
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 3 x 12 x 12
model.add(Conv2D('conv2', 3, 9, 5, 0, 1)) # output shape: N x 9 x 8 x 8
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 9 x 4 x 4
model.add(Reshape('flatten', (-1, 144)))
model.add(Linear('fc3', 144, 10, 0.1))

'''
# input: N x 1 x 28 x 28
model.add(Conv2D('conv1', 1, 4, 5, 1, 0.01)) # output shape: N x 4 x 26 x 26
model.add(Relu('relu1'))

model.add(Conv2D('conv1', 4, 1, 5, 0, 0.01)) # output shape: N x 1 x 22 x 22 
model.add(Relu('relu1'))

model.add(Reshape('flatten', (-1, 484)))

model.add(Linear('fc3', 484, 10, 0.1))
'''

'''
model.add(Reshape('flatten', (-1, 784)))
model.add(Linear('fc3', 784, 300, 0.1))
model.add(Relu('relu2'))
model.add(Linear('fc3', 300, 10, 0.1))
'''

loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.1,
    'weight_decay': 0,
    'momentum': 0.0,
    'batch_size': 50,
    'max_epoch': 10,
    'disp_freq': 50,
    'test_epoch': 1
}

acc_file = "btest_result.txt"
loss_file = "btrain_result.txt"

weight_file = "bweight.json"
bias_file = "bbias.json"
output_file = "boutput.json"

da_flag = False

cnt = 1
while cnt + 1 < len(sys.argv):
    opt = sys.argv[cnt]
    con = sys.argv[cnt + 1]
    cnt = cnt + 2
    if opt == "-lr":
        config['learning_rate'] = float(con)
	weight_file = "lr" + con + weight_file
	bias_file = "lr" + con + bias_file
	output_file = "lr" + con + output_file
    elif opt == '-wd':
        config['weight_decay'] == float(con)
	weight_file = "wd" + con + weight_file
	bias_file = "wd" + con + bias_file
	output_file = "wd" + con + output_file
    elif opt == '-bs':
        config['batch_size'] == int(con)
	weight_file = "bs" + con + weight_file
	bias_file = "bs" + con + bias_file
	output_file = "bs" + con + output_file
    elif opt == '-test':
        acc_file == con
    elif opt == '-train':
        loss_file == con
    elif opt == '-da':
        if con == 'on':
	    weight_file = "da" + con + weight_file
	    bias_file = "da" + con + bias_file
	    output_file = "da" + con + output_file
	    da_flag = True

# data augmentation
if da_flag:
    temp_data = train_data.copy()
    temp_label = train_label.copy()

    N = train_data.shape[0]
    
    for i in range(1):
        train_data = np.append(train_data, temp_data, axis=0)
        train_label = np.append(train_label, temp_label, axis=0)
    
    for n in range(N, 2*N):
        # image = np.reshape(train_data[n], (28, 28))
        image = misc.imrotate(train_data[n][0], 10*np.random.randn()) / 255.0
        train_data[n][0] = image
        
    
os.system("rm " + acc_file)
os.system("touch " + acc_file)
os.system("rm " + loss_file)
os.system("touch " + loss_file)

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))

    lw = conv1.W.tolist()
    with open(weight_file, 'w') as f:
    	json.dump(lw, f)

    lb = conv1.b.tolist()
    with open(bias_file, 'w') as f:
    	json.dump(lb, f)

    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], loss_file)

    lo = relu1.output.tolist()
    with open(output_file, 'w') as f:
    	json.dump(lo, f)

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        los, accu = test_net(model, loss, test_data, test_label, config['batch_size'])

	outf = file(acc_file, "a")
        outf.write(str(epoch) + ' ' + str(los) + ' ' + str(accu) + ' ' + str(time.time()) + '\n')
	outf.close()
