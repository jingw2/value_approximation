#!usr/bin/python 3.6
#!-*-coding:utf-8-*-

'''
utility functions
'''

import skimage.measure
import numpy as np
import random
import torch
import cv2

def preprocess(img):
	'''
	preprocess image 

	Args:
	img (np.array): (210, 160, 3)
	'''
	# img = img[20:230, :, :]

	# to grayscale
	grayscale = np.dot(img, [0.2989, 0.5870, 0.1140])

	# downsample
	# downsample = grayscale[::2, ::2]
	downsample = cv2.resize(grayscale, (84, 84))

	# normalized
	normalize = downsample / 255. # (105, 80)

	# channel first
	normalize = torch.Tensor(normalize).unsqueeze(0).unsqueeze(1)

	return normalize

def scale_reward(reward):
	'''
	scale reward to -1, 0, 1
	Args:
	reward (float)
	'''
	return np.sign(reward)


class ReplayMemory(object):
    
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
    
    def push(self, transition):
        if len(self.memory) == self.buffer_size:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def empty(self):
        self.memory = []