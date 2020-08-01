import logging
import pdb
import glob
import math
import argparse
import time
from datetime import datetime
import io, sys, os
import gc
import _pickle as pickle
import base64
import argparse

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
from neighborhood_v4_ddqn.models import *
from neighborhood_v4_ddqn.utils import *

# === Set up environment ===
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def eval(agent, env, train_configs, env_configs, ep_ids_to_skip, device, log_name, 
         episode_lengths=False, collision_ep_agents_init=None, 
         collision_folder_name=None, during_training=False):
        
    ni_ep_len_dict = {1: 200, 2: 200, 3: 200}
    total_reward = 0
    total_episode_lengths = 0
    total_dist_driven = 0
    episode_score = 0
    episode_length = 0
    collision_cnt = 0
    success_cnt = 0 # success means ego reaches the end before timeout
    avg_velocity = 0

    # how many saved episodes you wanna use
    num_saved_episodes = len(collision_ep_agents_init)
    episodes_run = 0

    for episode in range(1, num_saved_episodes + 1):
        episodes_run += 1
        # decide whether to reset using saved episodes or start fresh
        # The caller should instantiate the env with the correct env_config. 
        # num_other_agents is different across episodes
        if episode <= num_saved_episodes and (collision_ep_agents_init is not None): # use collision_ep_agents_init
            ep_idx = episode - 1
            saved_env_config = collision_ep_agents_init[ep_idx]['env_config']
            loaded_agents = collision_ep_agents_init[ep_idx]['agents_init']
            ep_id = collision_ep_agents_init[ep_idx]['ep_id']

            if ep_id in ep_ids_to_skip:
                continue

            # adjust ni, max_na, max_ep_len based on saved episode
            env.env_config_['ego_num_intersections_in_path'] = saved_env_config['ego_num_intersections_in_path']
            env.env_config_['max_num_other_agents'] = saved_env_config['max_num_other_agents']
            env.env_config_['max_episode_timesteps'] = ni_ep_len_dict[saved_env_config['ego_num_intersections_in_path']]
            train_configs['max_episode_timesteps'] = env.env_config_['max_episode_timesteps']
            env.env_config_['agent_action_noise'] = saved_env_config['agent_action_noise']
            log(log_name, 'Updated env_config: ' + str(env.env_config_))
            
            # the saved env config for each episode should have these fields coincide with the env config during eval
            for key in ['agent_stochastic_stop', 'agent_shuffle_ids',
                    'expanded_lane_set_depth', 'c1', 'ego_expand_path_depth', 
                    'single_intersection_type','ego_velocity_coeff','agent_velocity_coeff']: # 'ego_expand_path_depth' and 'c1' not tested for 0313 eval sets
                if ((key not in env.env_config_) and (key in saved_env_config)) or \
                ((key in env.env_config_) and (key not in saved_env_config)) or \
                (env.env_config_[key] != saved_env_config[key]):
                    raise Exception(f'eval env_config key mismatch {key}. saved: {saved_env_config[key]}. env: {env.env_config_[key]}')

            # add this check to avoid the issue of incorrect save env config
            if len(loaded_agents) != (saved_env_config['num_other_agents'] + 1):
                env.env_config_['num_other_agents'] = len(loaded_agents) - 1
            else:
                env.env_config_['num_other_agents'] = saved_env_config['num_other_agents']
            parametric_state, ego_b1 = env.reset(use_saved_agents=loaded_agents)

            # debug
            num_other_agents = env.env_config_['num_other_agents']
            max_num_other_agents = env.env_config_['max_num_other_agents']

            record_str = f"[Eval Episode {episode} | episodes_run={episodes_run} | ep_id={ep_id}] num other agents: {num_other_agents}. max_num_other_agents: {max_num_other_agents}"
            log(log_name, record_str)
            if not during_training:
                print(record_str)
        else:
            parametric_state, ego_b1 = env.reset()
            log(log_name, f"[Eval episode {episode}] Fresh reset")
        
        done = False
        episode_length = 0
        episode_score = 0

        while (done == False) and (episode_length < env.env_config_['max_episode_timesteps']):
            # select action
            if train_configs['num_future_states'] > 0:
                parametric_state_till_now = truncate_state_till_now(parametric_state, env_configs)
            else:
                parametric_state_till_now = parametric_state
            parametric_state_ts = torch.from_numpy(parametric_state_till_now).unsqueeze(0).float().to(device)
            action = agent.select_action(parametric_state_ts, ego_b1, 0, test=True)
            next_state, reward, done, info = env.step(action) # step processed action
            action_str = "selected action: " + str(action)
            log(log_name, f'[Eval episode {episode} ts={episode_length}] reward={reward:.1f} | {action_str}')

            parametric_state = next_state

            episode_score += reward
            episode_length += 1
        
        if (info[1] == True):
            collision_cnt += 1
            log(log_name, f"[Eval episode {episode} len={episode_length}] episode score={episode_score:.1f} | episode_dist_driven = {info[2]} | collision")
            if not during_training:
                print(f"[Eval episode {episode} len={episode_length}] episode score={episode_score:.1f} | episode_dist_driven = {info[2]} | collision")
            if collision_folder_name is not None:
                if not os.path.exists(f'collision/{collision_folder_name}'):
                    os.mkdir(f'collision/{collision_folder_name}')
                if not os.path.exists(f'collision/{collision_folder_name}/{episode}'):
                    os.mkdir(f'collision/{collision_folder_name}/{episode}')
                env.render(f'collision/{collision_folder_name}/{episode}')
        elif done == True:
            success_cnt += 1
            log(log_name, f"[Eval episode {episode} len={episode_length}] episode score={episode_score:.1f} | episode_dist_driven = {info[2]} | success")
            if not during_training:
                print(f"[Eval episode {episode} len={episode_length}] episode score={episode_score:.1f} | episode_dist_driven = {info[2]} | success")
        else:
            log(log_name, f"[Eval episode {episode} len={episode_length}] episode score={episode_score:.1f} | episode_dist_driven = {info[2]} | timeout")
            if not during_training:
                print(f"[Eval episode {episode} len={episode_length}] episode score={episode_score:.1f} | episode_dist_driven = {info[2]} | timeout")

        total_reward += episode_score
        avg_velocity += info[0]
        total_episode_lengths += episode_length
        total_dist_driven += info[2]

    avg_score = total_reward / episodes_run
    avg_velocity = avg_velocity / episodes_run
    collision_pct = collision_cnt / episodes_run * 100
    success_pct = success_cnt / episodes_run * 100
    avg_episode_lengths = total_episode_lengths / episodes_run
    if collision_cnt == 0:
        km_per_collision = total_dist_driven / 1000.0
    else:
        km_per_collision = (total_dist_driven / 1000.0) / collision_cnt
    
    if episode_lengths:
        return avg_score, collision_pct, success_pct, avg_velocity, km_per_collision, avg_episode_lengths
    else:
        return avg_score, collision_pct, success_pct, avg_velocity, km_per_collision
