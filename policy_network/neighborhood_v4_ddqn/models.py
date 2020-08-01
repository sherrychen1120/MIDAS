import logging
import pdb
import glob
import math
import time
import random
from datetime import datetime
import io, sys, os, copy
import base64
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from gym import logger as gymlogger
from gym.wrappers import Monitor
import gym
import gym_road_interactions

from neighborhood_v4_ddqn.utils import ReplayBuffer, log
from neighborhood_v4_ddqn.modules import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class DQN(nn.Module):
    """
    Network that has a backbone MLP and a head for value estimation for each action
    """
    def __init__(self, train_configs):
        super(DQN, self).__init__()
        self.train_configs = train_configs

        self.s_encoder = nn.Sequential(
            nn.Linear(self.train_configs['state_dim'], 128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        self.b1_fc = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,128)
        )

        self.int_layer = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        # Value estimation
        self.v_fc = nn.Linear(128, self.train_configs['action_dim'])

    def forward(self, s, b1):
        h = self.s_encoder(s).view(s.size(0),-1) # bsize*128
        z_b1 = self.b1_fc(b1) # bsize*128
        z = self.int_layer(h + z_b1) # bsize*128
        # value
        v = self.v_fc(z)
        return v

class TwinDQN(nn.Module):
    """
    Network that has a backbone MLP and 2 heads for value estimations
    """
    def __init__(self, train_configs, input_size=None):
        super(TwinDQN, self).__init__()
        self.train_configs = train_configs

        if input_size is None:
            self.input_size = self.train_configs['agent_total_state_dim'] * (self.train_configs['max_num_other_agents_in_range'] + 1) 
        else:
            self.input_size = input_size

        self.enc = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        self.b1_fc = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,128)
        )

        self.v_fc1 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, self.train_configs['action_dim'])
        )

        self.v_fc2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, self.train_configs['action_dim'])
        )
        
    def forward(self, o, b1):
        z_o = self.enc(o).view(o.size(0),-1) # bsize*128
        z_b1 = self.b1_fc(b1) # bsize*128
        q1 = self.v_fc1(z_o + z_b1)
        q2 = self.v_fc2(z_o + z_b1)
        return q1, q2
    
    def Q1(self, o, b1):
        z_o = self.enc(o).view(o.size(0),-1) # bsize*128
        z_b1 = self.b1_fc(b1) # bsize*128
        q1 = self.v_fc1(z_o + z_b1)
        return q1
    
    def encode(self, o, b1):
        z_o = self.enc(o).view(o.size(0),-1) # bsize*128
        z_b1 = self.b1_fc(b1) # bsize*128
        return (z_o + z_b1) # bsize*128
    
    def heads(self, h):
        # h: bsize*128
        q1 = self.v_fc1(h)
        q2 = self.v_fc2(h)
        return q1, q2
    
    def head_Q1(self, h):
        q1 = self.v_fc1(h)
        return q1

class DeepSet(nn.Module):
    """
    Network that uses MLP to encode every agent state and combine them using mean or max pooling
    and a head for value estimation for each action
    """
    def __init__(self, train_configs):
        super(DeepSet, self).__init__()
        self.train_configs = train_configs

        # encode each agent state vector
        self.s_encoder = nn.Sequential(
            nn.Linear(self.train_configs['agent_total_state_dim'], 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,128)
        )

        self.b1_fc = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,128)
        )

        self.int_layer = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        # value estimation using the mean or max-pooled state vector
        self.v_fc1 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, self.train_configs['action_dim'])
        )

    def forward(self, s, b1):
        # s: bsize * state_dim
        s = s.view(s.size(0), -1, self.train_configs['agent_total_state_dim']) # bsize * max_num_agents * agent_state_dim
        z = self.s_encoder(s) # bsize * max_num_agents * 64
        z_b1 = self.b1_fc(b1).unsqueeze(1) # bsize * 1 * 128
        z_0 = z[:,0,:].unsqueeze(1) + z_b1
        z = torch.cat([z_0, z[:,1:,:]], dim=1) # bsize * max_num_agents * 128
        z = self.int_layer(z) # bsize * max_num_agents * 128
        if self.train_configs['pooling'] == 'mean':
            z = z.mean(-2) # bsize * 128
        elif self.train_configs['pooling'] == 'max':
            z = z.max(-2)[0] # bsize * 128
        # value
        q1 = self.v_fc1(z)
        return q1

class TwinDeepSet(nn.Module):
    """
    Network that uses MLP to encode every agent state and combine them using mean or max pooling
    and a head for value estimation for each action
    """
    def __init__(self, train_configs):
        super(TwinDeepSet, self).__init__()
        self.train_configs = train_configs

        # encode each agent state vector
        self.s_encoder = nn.Sequential(
            nn.Linear(self.train_configs['agent_state_dim'], 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,128)
        )

        self.b1_fc = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,128)
        )

        self.int_layer = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        # value estimation using the mean or max-pooled state vector
        self.v_fc1 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, self.train_configs['action_dim'])
        )

        self.v_fc2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, self.train_configs['action_dim'])
        )

    def forward(self, s, b1):
        # s: bsize * state_dim
        s = s.view(s.size(0), -1, self.train_configs['agent_state_dim']) # bsize * max_num_agents * agent_state_dim
        z = self.s_encoder(s) # bsize * max_num_agents * 64
        z_b1 = self.b1_fc(b1).unsqueeze(1) # bsize * 1 * 128
        z_0 = z[:,0,:].unsqueeze(1) + z_b1
        z = torch.cat([z_0, z[:,1:,:]], dim=1) # bsize * max_num_agents * 128
        z = self.int_layer(z) # bsize * max_num_agents * 128
        if self.train_configs['pooling'] == 'mean':
            z = z.mean(-2) # bsize * 64
        elif self.train_configs['pooling'] == 'max':
            z = z.max(-2)[0] # bsize * 64
        # value
        q1 = self.v_fc1(z)
        q2 = self.v_fc1(z)
        return q1, q2
    
    def Q1(self, s, b1):
        # s: bsize * state_dim
        s = s.view(s.size(0), -1, self.train_configs['agent_state_dim']) # bsize * max_num_agents * agent_state_dim
        z = self.s_encoder(s) # bsize * max_num_agents * 64
        z_b1 = self.b1_fc(b1).unsqueeze(1) # bsize * 1 * 128
        z_0 = z[:,0,:].unsqueeze(1) + z_b1
        z = torch.cat([z_0, z[:,1:,:]], dim=1) # bsize * max_num_agents * 128
        z = self.int_layer(z) # bsize * max_num_agents * 128
        if self.train_configs['pooling'] == 'mean':
            z = z.mean(-2) # bsize * 128
        elif self.train_configs['pooling'] == 'max':
            z = z.max(-2)[0] # bsize * 128
        # value
        q1 = self.v_fc1(z)
        return q1

class SetTransformer(nn.Module):
    def __init__(self, train_configs, input_size=None):
        super(SetTransformer, self).__init__()
        self.train_configs = train_configs
        ln = train_configs['layer_norm']
        dim_hidden = 128
        self.dim_hidden = dim_hidden
        num_heads = 4
        num_inds = 32

        if input_size is None:
            self.input_size = self.train_configs['agent_total_state_dim']
        else:
            self.input_size = input_size

        self.b1_fc = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        if ('layers' in list(self.train_configs.keys())) and (self.train_configs['layers'] == 1):
            self.enc = nn.Sequential(
                    ISAB(dim_in=self.input_size, dim_out=dim_hidden, 
                        num_heads=num_heads, num_inds=num_inds, ln=ln))
            self.dec = nn.Sequential(
                PMA(dim=dim_hidden, num_heads=num_heads, 
                    num_seeds=self.train_configs['action_dim'], ln=ln),
                SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                nn.Linear(dim_hidden, 1))
            if ('model' in list(self.train_configs.keys())) and (self.train_configs['model'] == 'TwinDDQN'):
                self.dec2 = nn.Sequential(
                    PMA(dim=dim_hidden, num_heads=num_heads, 
                        num_seeds=self.train_configs['action_dim'], ln=ln),
                    SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                    nn.Linear(dim_hidden, 1))
        else:
            self.enc = nn.Sequential(
                    ISAB(dim_in=self.input_size, dim_out=dim_hidden, 
                        num_heads=num_heads, num_inds=num_inds, ln=ln),
                    ISAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=ln))
            self.dec = nn.Sequential(
                PMA(dim=dim_hidden, num_heads=num_heads, 
                    num_seeds=self.train_configs['action_dim'], ln=ln),
                SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                nn.Linear(dim_hidden, 1))
            if ('model' in list(self.train_configs.keys())) and (self.train_configs['model'] == 'TwinDDQN'):
                self.dec2 = nn.Sequential(
                    PMA(dim=dim_hidden, num_heads=num_heads, 
                        num_seeds=self.train_configs['action_dim'], ln=ln),
                    SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                    SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
                    nn.Linear(dim_hidden, 1))

    def forward(self, s, b1):
        # s: bsize * state_dim
        # b1: bsize * 1
        s = s.view(s.size(0), -1, self.input_size)
        z = self.enc(s)
        z_b1 = self.b1_fc(b1).unsqueeze(1)
        z_0 = z[:,0,:].unsqueeze(1) + z_b1
        z = torch.cat([z_0, z[:,1:,:]], dim=1) # bsize * max_num_agents * agent_state_dim
        q1 = self.dec(z).view(s.size(0), self.train_configs['action_dim']) # bsize * 2
        if ('model' in list(self.train_configs.keys())) and (self.train_configs['model'] == 'TwinDDQN'):
            q2 = self.dec2(z).view(s.size(0), self.train_configs['action_dim']) # bsize * 2
            return q1, q2
        else:
            return q1

    def Q1(self, s, b1):
        # s: bsize * state_dim
        # b1: bsize * 1
        s = s.view(s.size(0), -1, self.input_size)
        z = self.enc(s)
        z_b1 = self.b1_fc(b1).unsqueeze(1)
        z_0 = z[:,0,:].unsqueeze(1) + z_b1
        z = torch.cat([z_0, z[:,1:,:]], dim=1) # bsize * max_num_agents * agent_state_dim
        q1 = self.dec(z).view(s.size(0), self.train_configs['action_dim']) # bsize * 2
        return q1

class SocialAttention(nn.Module):
    def __init__(self, train_configs):
        super(SocialAttention, self).__init__()
        self.train_configs = train_configs
        dim_hidden = 64
        num_heads = 2

        self.enc = nn.Sequential(
                nn.Linear(self.train_configs['agent_state_dim'], dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))

        self.b1_fc = nn.Sequential(
            nn.Linear(1,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_hidden)
        )

        self.int_layer = nn.Sequential(
            nn.Linear(dim_hidden,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_hidden)
        )

        self.attention_module = EgoAttention(dim_hidden, num_heads)
        
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, self.train_configs['action_dim']))

    def forward(self, s, b1):
        # s: bsize * state_dim
        s = s.view(s.size(0), -1, self.train_configs['agent_state_dim'])
        ego_s = s[:,0,:].view(s.size(0),1,self.train_configs['agent_state_dim'])

        z_b1 = self.b1_fc(b1).unsqueeze(1) # bsize * dim_hidden

        h_ego_s = self.enc(ego_s)
        z_ego_s = h_ego_s + z_b1
        z_ego_s = self.int_layer(z_ego_s)

        h_s = self.enc(s)
        z_0 = h_s[:,0,:].unsqueeze(1) + z_b1
        z_s = torch.cat([z_0, h_s[:,1:,:]], dim=1) # bsize * max_num_agents * dim_hidden
        z_s = self.int_layer(z_s)
        
        rst = self.attention_module(z_ego_s, z_s)
        rst = self.dec(rst)
        
        return rst.view(s.size(0), self.train_configs['action_dim']) # bsize * 2

class TwinSocialAttention(nn.Module):
    def __init__(self, train_configs):
        super(TwinSocialAttention, self).__init__()
        self.train_configs = train_configs
        dim_hidden = 64
        num_heads = 2

        self.enc = nn.Sequential(
                nn.Linear(self.train_configs['agent_total_state_dim'], dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))

        self.b1_fc = nn.Sequential(
            nn.Linear(1,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_hidden)
        )

        self.int_layer = nn.Sequential(
            nn.Linear(dim_hidden,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_hidden)
        )

        self.attention_module = EgoAttention(dim_hidden, num_heads)

        self.dec1 = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, self.train_configs['action_dim']))
        self.dec2 = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, self.train_configs['action_dim']))

    def forward(self, s, b1):
        # s: bsize * state_dim
        s = s.view(s.size(0), -1, self.train_configs['agent_total_state_dim'])
        ego_s = s[:,0,:].view(s.size(0),1,self.train_configs['agent_total_state_dim'])

        z_b1 = self.b1_fc(b1).unsqueeze(1) # bsize * dim_hidden

        h_ego_s = self.enc(ego_s)
        z_ego_s = h_ego_s + z_b1
        z_ego_s = self.int_layer(z_ego_s)

        h_s = self.enc(s)
        z_0 = h_s[:,0,:].unsqueeze(1) + z_b1
        z_s = torch.cat([z_0, h_s[:,1:,:]], dim=1) # bsize * max_num_agents * dim_hidden
        z_s = self.int_layer(z_s)
        
        rst = self.attention_module(z_ego_s, z_s)

        q1 = self.dec1(rst).view(s.size(0), self.train_configs['action_dim'])
        q2 = self.dec2(rst).view(s.size(0), self.train_configs['action_dim'])
        
        return q1, q2
    
    def Q1(self, s, b1):
        # s: bsize * state_dim
        s = s.view(s.size(0), -1, self.train_configs['agent_total_state_dim'])
        ego_s = s[:,0,:].view(s.size(0),1,self.train_configs['agent_total_state_dim'])

        z_b1 = self.b1_fc(b1).unsqueeze(1) # bsize * dim_hidden

        h_ego_s = self.enc(ego_s)
        z_ego_s = h_ego_s + z_b1
        z_ego_s = self.int_layer(z_ego_s)

        h_s = self.enc(s)
        z_0 = h_s[:,0,:].unsqueeze(1) + z_b1
        z_s = torch.cat([z_0, h_s[:,1:,:]], dim=1) # bsize * max_num_agents * dim_hidden
        z_s = self.int_layer(z_s)
        
        rst = self.attention_module(z_ego_s, z_s)

        q1 = self.dec1(rst).view(s.size(0), self.train_configs['action_dim'])
        
        return q1

# TwinDDQN agent
class TwinDDQNAgent(object):
    def __init__(self, train_configs, device, log_name):
        self.steps_done = 0 # not concerned with exploration steps
        self.train_configs = train_configs
        self.replay_buffer = ReplayBuffer(int(train_configs['max_buffer_size']))
        self.log_name = log_name

        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        if 'use_lstm' in list(self.train_configs.keys()) and self.train_configs['use_lstm']:
            log(self.log_name, f"Using LSTMValueNet")
            self.value_net = LSTMValueNet(train_configs, self.device).to(self.device)
            self.value_net_target = LSTMValueNet(train_configs, self.device).to(self.device)
        else:
            if 'value_net' in list(self.train_configs.keys()):
                if self.train_configs['value_net'] == 'vanilla':
                    self.value_net = TwinDQN(train_configs).to(self.device)
                    self.value_net_target = TwinDQN(train_configs).to(self.device)
                elif self.train_configs['value_net'] == 'deep_set':
                    self.value_net = TwinDeepSet(train_configs).to(self.device)
                    self.value_net_target = TwinDeepSet(train_configs).to(self.device)
                elif self.train_configs['value_net'] == 'deeper_deep_set':
                    raise Exception('Not implemented')
                elif self.train_configs['value_net'] == 'set_transformer':
                    self.value_net = SetTransformer(train_configs).to(self.device)
                    self.value_net_target = SetTransformer(train_configs).to(self.device)
                elif self.train_configs['value_net'] == 'social_attention':
                    self.value_net = TwinSocialAttention(train_configs).to(self.device)
                    self.value_net_target = TwinSocialAttention(train_configs).to(self.device)
                else:
                    value_net_type = self.train_configs['value_net']
                    raise Exception(f'Invalid value_net type: {value_net_type}')
            else:
                self.value_net = TwinDQN(train_configs).to(self.device)
                self.value_net_target = TwinDQN(train_configs).to(self.device)
    
        self.value_net_target.load_state_dict(self.value_net.state_dict())
        self.value_net_target.eval()
        self.value_net_optim = optim.Adam(self.value_net.parameters(), lr=self.train_configs['lrt'])
        # self.value_net_optim = optim.RMSprop(self.value_net.parameters(), lr=self.train_configs['lrt'])

        # self.sm = nn.Softmax(dim=1) # softmax used when generating random actions

    def select_action(self, parametric_state, b1, total_timesteps, test=False):
        if test or (total_timesteps > self.train_configs['exploration_timesteps']):
             # parametric_state: 1*(state_dim), tensor
            parametric_state = parametric_state.to(self.device)
            b1 = torch.tensor([[b1]]).to(self.device) # (1,1)
            # v = self.value_net(parametric_state)

            sample = np.random.random()
            if test:
                eps_threshold = self.train_configs['eps_end']
            else:
                eps_threshold = self.train_configs['eps_end'] + (self.train_configs['eps_start'] - self.train_configs['eps_end']) * \
                    math.exp(-1. * total_timesteps / self.train_configs['eps_decay'])
            if sample > eps_threshold:
                with torch.no_grad():
                    # == V0, V1 ==
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    # action = self.value_net.Q1(parametric_state, b1).max(1)[1].item()

                    # == V2, V3, V4, V5 ==
                    Q1, Q2 = self.value_net(parametric_state, b1)
                    action = (Q1 + Q2).max(1)[1].item()

                    return action
            else:
                return np.random.randint(2)
        else:
            return np.random.randint(2)
    
    def __dropout_future_states(self, s, s_next):
        sample = np.random.uniform()
        if sample <= self.train_configs['future_state_dropout']:
            log(self.log_name, f"Dropout s, s_next")
            s = copy.deepcopy(s)
            s_next = copy.deepcopy(s_next)

            s = np.reshape(s, (-1, self.train_configs['num_ts_in_state'], self.train_configs['agent_state_dim']))
            s[:,-(self.train_configs['num_future_states']):,:] = 0
            s = np.reshape(s, (-1, self.train_configs['state_dim']))
            s_next = np.reshape(s_next, (-1, self.train_configs['num_ts_in_state'], self.train_configs['agent_state_dim']))
            s_next[:,-(self.train_configs['num_future_states']):,:] = 0
            s_next = np.reshape(s_next, (-1, self.train_configs['state_dim']))
        return s, s_next

    def train(self):
        # perform one model update
        # sample replay buffer
        #   batch_size*state_dim, batch_size*state_dim, 
        #   batch_size*action_dim, batch_size*1, batch_size*1
        # tensor
        s,s_next,a,r,d,b1 = self.replay_buffer.sample(self.train_configs['batch_size'])
        if self.train_configs['num_future_states'] > 0:
            s,s_next = self.__dropout_future_states(s,s_next)

        s = torch.from_numpy(s).float().to(self.device)
        s_next = torch.from_numpy(s_next).float().to(self.device)
        a = torch.from_numpy(a).to(self.device).view(-1,1)
        r = torch.from_numpy(r).float().to(self.device).view(-1,1)
        not_d = torch.from_numpy(1-d).float().to(self.device).view(-1,1)
        b1 = torch.from_numpy(b1).float().to(self.device).view(-1,1)

        # calculate Q(s_t,a_t)
        # pdb.set_trace()
        curr_Q1_values, curr_Q2_values = self.value_net(s, b1)
        curr_Q1 = curr_Q1_values.gather(1,a)
        curr_Q2 = curr_Q2_values.gather(1,a)

        next_Q1_values, next_Q2_values = self.value_net(s_next, b1)
        next_Q1_values_target, next_Q2_values_target = self.value_net_target(s_next, b1)

        # calculate Q(s_{t+1}, a')
        # ======= BRANCHES =======
        # == V0 ==
        # actions = next_Q1_values.max(1)[1].long().view(-1,1)
        # next_Q = torch.min(next_Q1_values_target.gather(1, actions), next_Q2_values_target.gather(1, actions))
        # # Compute the target of the current Q values
        # expected_Q = r + (self.train_configs['gamma'] * not_d * next_Q)
        # # Compute Huber loss
        # # loss = F.smooth_l1_loss(curr_Q, expected_Q)
        # # Detach variable from the current graph since we don't want gradients for next Q to propagated
        # loss = F.mse_loss(curr_Q1, expected_Q.detach()) + F.mse_loss(curr_Q2, expected_Q.detach())

        # == V1 ==
        # actions_1 = next_Q1_values.max(1)[1].long().view(-1,1)
        # actions_2 = next_Q2_values.max(1)[1].long().view(-1,1)
        # next_Q = torch.min(next_Q1_values_target.gather(1, actions_1), next_Q2_values_target.gather(1, actions_2))
        # # Compute the target of the current Q values
        # expected_Q = r + (self.train_configs['gamma'] * not_d * next_Q)
        # # Compute loss
        # loss = F.mse_loss(curr_Q1, expected_Q.detach()) + F.mse_loss(curr_Q2, expected_Q.detach())

        # == V2 ==
        actions_1 = next_Q1_values.max(1)[1].long().view(-1,1)
        actions_2 = next_Q2_values.max(1)[1].long().view(-1,1)
        next_Q = torch.min(next_Q1_values_target.gather(1, actions_2), next_Q2_values_target.gather(1, actions_1))
        # Compute the target of the current Q values
        expected_Q = r + (self.train_configs['gamma'] * not_d * next_Q)
        # Compute loss
        loss = F.mse_loss(curr_Q1, expected_Q.detach()) + F.mse_loss(curr_Q2, expected_Q.detach())

        # == V3 ==
        # actions = (next_Q1_values + next_Q2_values).max(1)[1].long().view(-1,1)
        # next_Q = torch.min(next_Q1_values_target.gather(1, actions), next_Q2_values_target.gather(1, actions))
        # # Compute the target of the current Q values
        # expected_Q = r + (self.train_configs['gamma'] * not_d * next_Q)
        # # Compute loss
        # loss = F.mse_loss(curr_Q1, expected_Q.detach()) + F.mse_loss(curr_Q2, expected_Q.detach())

        # == V3 ==
        # actions = (next_Q1_values + next_Q2_values).max(1)[1].long().view(-1,1)
        # next_Q = torch.min(next_Q1_values_target.gather(1, actions), next_Q2_values_target.gather(1, actions))
        # # Compute the target of the current Q values
        # expected_Q = r + (self.train_configs['gamma'] * not_d * next_Q)
        # # Compute loss
        # loss = F.mse_loss(curr_Q1, expected_Q.detach()) + F.mse_loss(curr_Q2, expected_Q.detach())

        # == V4 ==
        # actions_1 = next_Q1_values.max(1)[1].long().view(-1,1)
        # actions_2 = next_Q2_values.max(1)[1].long().view(-1,1)
        # next_Q_1 = next_Q1_values_target.gather(1, actions_2)
        # next_Q_2 = next_Q2_values_target.gather(1, actions_1)
        # # Compute the target of the current Q values
        # expected_Q_1 = r + (self.train_configs['gamma'] * not_d * next_Q_1)
        # expected_Q_2 = r + (self.train_configs['gamma'] * not_d * next_Q_2)
        # # Compute loss
        # loss = F.mse_loss(curr_Q1, expected_Q_1.detach()) + F.mse_loss(curr_Q2, expected_Q_2.detach())

        # == V5 ==
        # temp = np.random.random() # [0,1)
        # # UPDATE(A)
        # if temp < 0.5:
        #     actions_2 = next_Q2_values.max(1)[1].long().view(-1,1)
        #     next_Q_1 = next_Q1_values_target.gather(1, actions_2)
        #     expected_Q_1 = r + (self.train_configs['gamma'] * not_d * next_Q_1)
        #     loss = F.mse_loss(curr_Q1, expected_Q_1.detach())
        # # UPDATE(B)
        # else:
        #     actions_1 = next_Q1_values.max(1)[1].long().view(-1,1)
        #     next_Q_2 = next_Q2_values_target.gather(1, actions_1)
        #     expected_Q_2 = r + (self.train_configs['gamma'] * not_d * next_Q_2)
        #     loss = F.mse_loss(curr_Q2, expected_Q_2.detach())
        
        # == V6 ==
        # actions_1 = next_Q1_values.max(1)[1].long().view(-1,1)
        # actions_2 = next_Q2_values.max(1)[1].long().view(-1,1)
        # actions = (next_Q1_values + next_Q2_values).max(1)[1].long().view(-1,1)
        
        # next_Q_V1 = torch.min(next_Q1_values_target.gather(1, actions_1), next_Q2_values_target.gather(1, actions_2))
        # # Compute the target of the current Q values
        # expected_Q_V1 = r + (self.train_configs['gamma'] * not_d * next_Q_V1)
        # # Compute loss
        # loss_V1 = F.mse_loss(curr_Q1, expected_Q_V1.detach()) + F.mse_loss(curr_Q2, expected_Q_V1.detach())

        # next_Q_V2 = torch.min(next_Q1_values_target.gather(1, actions_2), next_Q2_values_target.gather(1, actions_1))
        # # Compute the target of the current Q values
        # expected_Q_V2 = r + (self.train_configs['gamma'] * not_d * next_Q_V2)
        # # Compute loss
        # loss_V2 = F.mse_loss(curr_Q1, expected_Q_V2.detach()) + F.mse_loss(curr_Q2, expected_Q_V2.detach())

        # next_Q_V3 = torch.min(next_Q1_values_target.gather(1, actions), next_Q2_values_target.gather(1, actions))
        # # Compute the target of the current Q values
        # expected_Q_V3 = r + (self.train_configs['gamma'] * not_d * next_Q_V3)
        # # Compute loss
        # loss_V3 = F.mse_loss(curr_Q1, expected_Q_V3.detach()) + F.mse_loss(curr_Q2, expected_Q_V3.detach())

        # # Compute the target of the current Q values
        # expected_Q_1_V4 = r + (self.train_configs['gamma'] * not_d * next_Q1_values_target.gather(1, actions_2))
        # expected_Q_2_V4 = r + (self.train_configs['gamma'] * not_d * next_Q2_values_target.gather(1, actions_1))
        # # Compute loss
        # loss_V4 = F.mse_loss(curr_Q1, expected_Q_1_V4.detach()) + F.mse_loss(curr_Q2, expected_Q_2_V4.detach())

        # loss = loss_V1 + loss_V2 + loss_V3 + loss_V4
        
        # ======= END OF BRANCHES =======

        # Optimize
        self.value_net.zero_grad()
        loss.backward()
        if self.train_configs['grad_clamp'] == True:
            for param in self.value_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.value_net_optim.step()

        # Delayed policy updates
        self.steps_done += 1
        if ((self.steps_done + 1) % self.train_configs['target_update'] == 0):
            # Update the frozen target models
            # self.value_net_target.load_state_dict(self.value_net.state_dict())
            if self.log_name is not None:
                log(self.log_name, f"[TwinDDQNAgent] Updating Target Network at steps_done = {self.steps_done}")
            for param, target_param in zip(self.value_net.parameters(), self.value_net_target.parameters()):
                target_param.data.copy_(self.train_configs['tau'] * param.data + (1 - self.train_configs['tau']) * target_param.data)

        return loss.item()
    
    def reduce_lrt(self, new_lrt : float):
        for param_group in self.value_net_optim.param_groups:
            param_group['lr'] = new_lrt

    def save(self, scores, file_path, info=None, env_configs=None, 
             train_configs=None, episode_lengths=None, eval_scores=None, 
             collision_pcts=None, avg_velocities=None, train_collisions=None, 
             eval_timeout_pcts=None, train_timeouts=None, eval_km_per_collision=None): # optional error info to save
        torch.save({'value_net': self.value_net.state_dict(),
                    'scores': scores,
                    'info': info,
                    'env_configs': env_configs,
                    'train_configs': train_configs,
                    'eval_scores': eval_scores, 
                    'collision_pcts': collision_pcts,
                    'avg_velocities': avg_velocities,
                    'episode_lengths': episode_lengths,
                    'train_collisions': train_collisions,
                    'eval_timeout_pcts': eval_timeout_pcts,
                    'train_timeouts': train_timeouts,
                    'eval_km_per_collision': eval_km_per_collision}, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.value_net_target.load_state_dict(checkpoint['value_net'])
        return checkpoint
