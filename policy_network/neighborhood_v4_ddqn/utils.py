import logging
import pdb
import glob
import math
import time
from datetime import datetime
import io, sys, os, copy
import base64

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

from gym import logger as gymlogger
from gym.wrappers import Monitor
import gym
import gym_road_interactions

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple replay buffer that saves a list of tuples of (state, next_state, action, reward, done) (tensors)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.cnt = 0

    def add(self, state, new_state, action, reward, done_bool, b1):
        # state: (state_dim,)
        # new_state: (state_dim,)
        # action: (action_dim,)
        # reward: scalar
        # done_bool: scalar
        # b1: scalar
        # numpy arrays

        data = (state, new_state, action, reward, done_bool, b1)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
            self.cnt += 1
    
    def sample(self, batch_size):
        # output:
        #   batch_size*state_dim, batch_size*state_dim, 
        #   batch_size*action_dim, batch_size*1, batch_size*1
        # np array
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s,ns,u,r,d,b1 = [],[],[],[],[],[]
        for i in ind:
            S, NS, U, R, D, B1 = self.storage[i]
            s.append(S)
            ns.append(NS)
            u.append(U)
            r.append(R)
            d.append(D)
            b1.append(B1)
        return np.array(s), np.array(ns), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), np.array(b1).reshape(-1, 1)
    
    def size(self):
        return self.cnt

# helper for calculating moving average of last 10 scores
def moving_average(scores, window):
    if len(scores) < window:
        return sum(scores) / len(scores) 
    else:
        return sum(scores[(-window):]) / float(window)

# Logging function
def log(fname, s):
    # if not os.path.isdir(os.path.dirname(fname)):
    #     os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    f.write(f'{str(datetime.now())}: {s}\n')
    f.close()

