from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime

import time

class CheckTime(object):
	
	conv_for = 0.0
	conv_bac = 0.0
	pool_for = 0.0 
	pool_bac = 0.0

	def init(self):
		self.conv_for = 0.0
		self.conv_bac = 0.0
		self.pool_for = 0.0
		self.pool_bac = 0.0
		self.rec = 0.0
	
	def goin(self):
		self.rec = int(round(time.time() * 1000)) / 1000.0
	
	def out_conv_for(self):
		temp = int(round(time.time() * 1000)) / 1000.0
		self.conv_for = self.conv_for + temp - self.rec
		
	def out_conv_bac(self):
		temp = int(round(time.time() * 1000)) / 1000.0
		self.conv_bac = self.conv_bac + temp - self.rec
		
	def out_pool_for(self):
		temp = int(round(time.time() * 1000)) / 1000.0
		self.pool_for = self.pool_for + temp - self.rec
		
	def out_pool_bac(self):
		temp = int(round(time.time() * 1000)) / 1000.0
		self.pool_bac = self.pool_bac + temp - self.rec
	
	def get(self):
		return self.conv_for, self.conv_bac, self.pool_for, self.pool_bac

check = CheckTime()
	
def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)
