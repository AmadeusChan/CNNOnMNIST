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

train_data, test_data, train_label, test_label = load_mnist_4d('data')
# train_data = train_data + np.random.randn(*train_data.shape) * 0.01

# Your model defintion here
# You should explore different model architecture
model = Network()

conv1 = Conv2D('conv1', 1, 6, 5, 0, 1) # output shape: N x 3 x 24 x 24
model.add(conv1)
relu1 = Relu('relu1')
model.add(relu1)
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 3 x 12 x 12
model.add(Conv2D('conv2', 6, 6, 5, 0, 1)) # output shape: N x 6 x 8 x 8
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 6 x 4 x 4
model.add(Reshape('flatten', (-1, 96)))
model.add(Linear('fc3', 96, 10, 0.1))

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
    'max_epoch': 100,
    'disp_freq': 10,
    'test_epoch': 1
}

acc_file = "test_result.txt"
loss_file = "train_result.txt"

os.system("rm " + acc_file)
os.system("touch " + acc_file)
os.system("rm " + loss_file)
os.system("touch " + loss_file)

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))

    lw = conv1.W.tolist()
    with open('weight.json', 'w') as f:
    	json.dump(lw, f)

    lb = conv1.b.tolist()
    with open('bias.json', 'w') as f:
    	json.dump(lb, f)

    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], loss_file)

    lo = relu1.output.tolist()
    with open('output.json', 'w') as f:
    	json.dump(lo, f)

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        los, accu = test_net(model, loss, test_data, test_label, config['batch_size'])

	outf = file(acc_file, "a")
        outf.write(str(epoch) + ' ' + str(los) + ' ' + str(accu) + ' ' + str(time.time()) + '\n')
	outf.close()
