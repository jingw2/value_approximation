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

def main():
	ENV = "Boxing-v0" # observation shape:  (250, 160, 3)
	env = gym.make(ENV).unwrapped
	global ACTION_SPACE, epsilon_start, epsilon_end, epsilon_decay
	global actor, target, optimizer, gamma, loss_fn, frame_num, use_ddqn
	ACTION_SPACE = env.action_space.n

	# hyperparameter
	n_epochs = 20
	buffer_size = 1000000
	epsilon_start = 1
	epsilon_end = 0.01
	epsilon_decay = 1000000
	replay_memory = util.ReplayMemory(buffer_size)
	# memory = pickle.load(open("replay_memory_18000.pkl", "rb"))
	# replay_memory.memory = memory
	# print("Current memory length: ", len(replay_memory))
	batch_size = 32
	frame_num = 4
	gamma = 0.99
	lr = 2.5e-4
	C = 100
	learning_starts = 10000
	learning_freq = 4
	use_ddqn = False

	actor = DQN(ACTION_SPACE, frame_num).to(device)
	target = deepcopy(actor).to(device)
	optimizer = RMSprop(actor.parameters(), lr = lr)
	loss_fn = nn.MSELoss()
	Transition = collections.namedtuple("transition", ("state", "action", "next_state", "reward"))

	# training
	running_rewards = []
	average_qvalue = []
	episode_rewards = []

	state = env.reset()
	frames = []
	if frame_num > 1:
		for _ in range(frame_num):
			frames.append(util.preprocess(state))
		state = torch.cat(frames, dim = 1).float()
	else:
		state = util.preprocess(state)

	reward_sum = 0
	done = False
	qvalue_sum = torch.zeros(batch_size, 1)
	num_param_update = 0
	for t in count():
		if t % 2000 == 0:
			# save replay memory
			# pickle.dump(replay_memory.memory, open("replay_memory_{}.pkl".format(t), "wb"))
			print("Finish step {}".format(t))
		epoch = t // learning_starts
		if t > learning_starts:
			action_value = actor(state)
			action = epsilon_greedy(action_value, t - learning_starts, env)
		else:
			action = env.action_space.sample()

		state_buffer = []
		reward_buffer = []
		for _ in range(frame_num):
			next_state, reward, done, _ = env.step(action)
			state_buffer.append(util.preprocess(next_state))
			reward_sum += reward
			reward_buffer.append(util.scale_reward(reward))
			if done:
				break
		next_state = torch.cat(state_buffer, dim = 1).float()
		
		if done:
			next_state = None          

		replay_memory.push(Transition(state, action, next_state, sum(reward_buffer)))

		if not done:
			state = next_state

		# train
		if t > learning_starts and t % learning_freq == 0 and len(replay_memory) > batch_size:
			qvalue = train(replay_memory, batch_size)
			average_qvalue.append((qvalue).mean().item())
			episode_rewards.append(reward_sum)
			if len(episode_rewards) > 0:
				average_episode_reward = np.mean(episode_rewards[-100:])
			print("Epoch {}, Step {}, Average Q value {}, Average episode reward {}".format(epoch, t, \
				average_qvalue[-1], average_episode_reward))
			num_param_update += 1

		# reset game
		if done:
			state = env.reset()
			frames = []
			if frame_num > 1:
				for _ in range(frame_num):
					frames.append(util.preprocess(state))
				state = torch.cat(frames, dim = 1).float().to(device)
			else:
				state = util.preprocess(state).to(device)
			reward_sum = 0

		# validation
		if num_param_update % C == 0:
			target = deepcopy(actor).to(device)		

		if epoch == n_epochs:
			break
		# if t > 1e6:
		# 	break

		# episode rewards
		plt.subplot(121)
		plt.plot(range(len(episode_rewards)), episode_rewards, "b-")
		plt.xlabel("step")
		plt.ylabel("reward")

		# q value
		plt.subplot(122)
		plt.plot(range(len(average_qvalue)), average_qvalue, "r-")
		plt.xlabel("step")
		plt.ylabel("qvalue")
		plt.savefig("episode_qvalue.png")

if __name__ == '__main__':
	main()
	