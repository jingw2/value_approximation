#!usr/bin/python 3.6
#!-*-coding:utf-8-*-

'''
Deep Q Network
Author: Jing Wang (jingwang@sf-express.com)
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import os
import gym
import collections
from PIL import Image
import util
from itertools import count
import pickle
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from torch.optim import lr_scheduler
import torchvision.transforms as T
import skimage.measure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN(nn.Module):
	
	def __init__(self, action_space, frame_num):
		super().__init__()
		# input should be (Batch, Channels, Height, Width)
		self.conv1 = nn.Conv2d(frame_num, 16, kernel_size = 8, stride = 4)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
		self.relu = nn.ReLU()
		self.affine1 = nn.Linear(2592, 256)
		self.affine2 = nn.Linear(256, action_space)
	
	def forward(self, x):
		'''
		Args:
		x (batch_size, 3, 250, 160)
		
		Return:
		prob
		'''
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = x.view(x.size(0), -1)
		x = self.relu(self.affine1(x))
		x = self.affine2(x)
		return x

def epsilon_greedy(action_value, step, env):
	'''
	epsilon greedy algorithm
	Args:
	action_value (tensor): (batch_size, action_space)
	step (int)
	env (gym.env)

	Return:
	action (int)
	'''
	epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * step / epsilon_decay)
	if random.random() > epsilon:
		with torch.no_grad():
			return torch.argmax(action_value, dim = 1).item()
	else:
		return env.action_space.sample()

def train(replay_memory, batch_size):

	if len(replay_memory) < batch_size:
		return 

	batch = replay_memory.sample(batch_size)
	
	states = Variable(torch.cat([b.state for b in batch], dim = 0).float()).to(device)
	actions = Variable(torch.cat([torch.Tensor([b.action], device = device).long() for b in batch]).view(-1))
	rewards = Variable(torch.cat([torch.Tensor([b.reward], device = device).float() for b in batch]).view(-1))
	
	not_end_states = torch.cat([b.next_state \
								for b in batch if b.next_state is not None], dim = 0).to(device)
	not_end_index = torch.tensor(tuple(map(lambda b: b.next_state is not None, \
										   batch)),device = device, dtype = torch.uint8)

	qvalue = actor(states).gather(1, actions.view(-1, 1))
	next_qvalue = torch.zeros(batch_size, device = device)
	if use_ddqn:
		next_qvalue[not_end_index] = target(not_end_states).gather(1, actions.view(-1, 1))
	else:
		next_qvalue[not_end_index] = target(not_end_states).max(dim = 1)[0].detach()
	y = rewards + gamma * next_qvalue
	loss = loss_fn(qvalue, y.unsqueeze(1))

	optimizer.zero_grad()
	loss.backward()
	torch.nn.utils.clip_grad_norm_(actor.parameters(), 1, norm_type=2)
	optimizer.step()

	return qvalue

def get_cart_location():
	world_width = env.x_threshold * 2
	scale = screen_width / world_width
	return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
	screen = env.render(mode='rgb_array').transpose(
		(2, 0, 1))  # transpose into torch order (CHW)
	# Strip off the top and bottom of the screen
	screen = screen[:, 160:320]
	view_width = 320
	cart_location = get_cart_location()
	if cart_location < view_width // 2:
		slice_range = slice(view_width)
	elif cart_location > (screen_width - view_width // 2):
		slice_range = slice(-view_width, None)
	else:
		slice_range = slice(cart_location - view_width // 2,
							cart_location + view_width // 2)
	# Strip off the edges, so that we have a square image centered on a cart
	screen = screen[:, :, slice_range]
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	# Convert to float, rescare, convert to torch tensor
	# (this doesn't require a copy)
	screen = screen.transpose((1, 2, 0))
	screen = cv2.resize(screen, (84, 84))
	screen = screen.transpose((2, 0, 1))
	screen = torch.Tensor(screen).float().unsqueeze(0)
	return screen

def main():

	global ACTION_SPACE, epsilon_start, epsilon_end, epsilon_decay, env, screen_width
	global actor, target, optimizer, gamma, loss_fn, frame_num, use_ddqn
	ENV = "CartPole-v0" # observation shape:  (250, 160, 3)
	env = gym.make(ENV).unwrapped
	ACTION_SPACE = env.action_space.n
	screen_width = 600

	# hyperparameter
	n_episodes = 2000
	buffer_size = 1000000
	epsilon_start = 1
	epsilon_end = 0.01
	epsilon_decay = 1000
	replay_memory = util.ReplayMemory(buffer_size)
	# memory = pickle.load(open("replay_memory_18000.pkl", "rb"))
	# replay_memory.memory = memory
	# print("Current memory length: ", len(replay_memory))
	batch_size = 128
	frame_num = 3
	gamma = 0.99
	lr = 1e-3
	C = 10
	use_ddqn = False

	actor = DQN(ACTION_SPACE, frame_num).to(device)
	target = deepcopy(actor).to(device)
	optimizer = RMSprop(actor.parameters(), lr = lr)
	loss_fn = nn.MSELoss()
	Transition = collections.namedtuple("transition", ("state", "action", "next_state", "reward"))

	# training
	episode_rewards = []
	average_qvalue = []
	for episode in range(n_episodes):
		env.reset()
		last_screen = get_screen()
		current_screen = get_screen()
		state = current_screen - last_screen
		done = False
		reward_sum = 0
		qvalue_sum = torch.zeros(ACTION_SPACE)
		while not done:
			action_value = actor(state)
			action = epsilon_greedy(action_value, episode, env)
			_, reward, done, _ = env.step(action)
			if not done:
				last_screen = get_screen()
				current_screen = get_screen()
				next_state = current_screen - last_screen
			else:
				next_state = None
			replay_memory.push(Transition(state, action, next_state, reward))
			if not done:
				state = next_state
			qvalue = train(replay_memory, batch_size)
			reward_sum += reward
			if qvalue is not None:
				qvalue_sum = qvalue_sum + qvalue

		episode_rewards.append(reward_sum)
		mean_qvalue = qvalue_sum.mean().item()
		average_qvalue.append(mean_qvalue)
		if episode % 100 == 0:
			print("Episode {}, last 100 episode rewards {}, total average rewards {}".format(episode, \
					np.mean(episode_rewards[-100:]), np.mean(episode_rewards)))
			print("Episode {}, last 100 episode qvalue {}, total average rewards {}".format(episode, \
					np.mean(average_qvalue[-100:]), np.mean(average_qvalue)))
		if episode % C == 0:
			target = deepcopy(actor).to(device)

		plt.subplot(121)
		plt.plot(range(len(episode_rewards)), episode_rewards, "b-")
		plt.xlabel("episode")
		plt.ylabel("reward")

		# q value
		plt.subplot(122)
		plt.plot(range(len(average_qvalue)), average_qvalue, "r-")
		plt.xlabel("episode")
		plt.ylabel("qvalue")
		plt.savefig("episode_qvalue_cartpole.png")

	env.close()

if __name__ == '__main__':
	main()
