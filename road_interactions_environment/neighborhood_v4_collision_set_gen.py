import gym
import gym_road_interactions
import time
import logging
import sys, os
import pdb
import glob
import copy, pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Logging function
def log(fname, s):
    f = open(fname, 'a')
    f.write(f'{str(datetime.now())}: {s}\n')
    f.close()

# format: [(model_path, ratio_of_num_other_agents)]
rl_agent_configs = [('',0.0)]

ni_ep_len_dict = {1: 200, 2:200, 3:200}
ni = 1
date = '0321'
env_configs = {# parametric state
              'use_global_frame': False, # whether to use global frame for the state vector
              'normalize': True, # whether to normalize the state vector
              'include_ttc': True,
              'include_future_waypoints': 10,
              'use_future_horizon_positions': False, # if false, just use the 10 future waypoints (each 10 waypoints apart)
              'num_history_states': 10, # number of history states included in parametric state. 
              'num_future_states': 10, # number of future states included in parametric state. 
              'time_gap': 1, # time gap between two states.
                             # total number of states included is: num_history_states + num_future_states + 1
                             # t - time_gap * n_h, t - time_gap * (n_h-1), ..., t, t + time_gap, ..., t + time_gap * n_f
              'stalemate_horizon': 5, # number of past ts (including current) that we track to determine stalemate
              'include_polygon_dist': True, # if true, include sigmoid(polygon distance between ego and agent)
              'include_agents_within_range': 10.0, # the range within which agents will be included.
              'agent_state_dim': 6+4+10*2+1, # state dim at a single ts. related to 'include_ttc', 'include_polygon_dist', 'include_future_waypoints''
              # num agents (change training env_config with this)
              'num_other_agents': 0, # this will change
              'max_num_other_agents': 25, # could only be 25, 40, 60 (one of the upper limits)
              'max_num_other_agents_in_range': 25, # >=6. max number of other agents in range. Must be <= max_num_other_agents. default 25.
              # agent behavior
              'agent_stochastic_stop': False, # (check) whether the other agent can choose to stop for ego with a certain probability
              'agent_shuffle_ids': True, # (check) if yes, the id of other agents will be shuffled during reset
              'rl_agent_configs' : [('',0.0)],
              'all_selfplay': False, # if true, all agents will be changed to the most recent rl model, whatever the episode mix is
              # path
              'ego_num_intersections_in_path': ni, # (check)
              'ego_expand_path_depth': 2, # (check) the number of extending lanes from center intersection in ego path
              'expanded_lane_set_depth': 1, # (check)
              'single_intersection_type': 'mix', # (check) ['t-intersection', 'roundabout', 'mix']
              'c1': 2, # (check) 0: left/right turn at t4, 1: all possible depth-1 paths at t4, 2: all possible depth-1 paths at random intersection
              # training settings
              'gamma': 0.99,
              'max_episode_timesteps': ni_ep_len_dict[ni],
              # NOTE: if you change velocity_coeff, you have to change the whole dataset 
              # (interaction set, collision set, etc) bc they are based on b1 and locations are 
              # based on v_desired
              'ego_velocity_coeff': (2.7, 8.3), # (check) (w, b). v_desired = w * b1 + b # experiments change this one
              'agent_velocity_coeff': (2.7, 8.3), # (check) (w, b). v_desired = w * b1 + b # this is fixed!
              # reward = w * b1 + b # Be careful with signs!
              'reward': 'default_fad', # [default, default_ttc, simplified, default_fad]
              'time_penalty_coeff': (-1./20., -3./20.),
              'velocity_reward_coeff': (0.5, 1.5),
              'collision_penalty_coeff': (-5.0, -45.0),
              'fad_penalty_coeff': 1.0,
              'timeout_penalty_coeff': (-5.0, -20.0),
              'stalemate_penalty_coeff': (-0.5, -1.5),
              # ego config
              'use_default_ego': False,
              'no_ego': False, # TODO: remember to turn this off if you want ego!
              # action noises
              'ego_action_noise': 0.0, # lambda of the poisson noise applied to ego action
              'agent_action_noise': 0.0, # lambda of the poisson noise applied to agent action
              # added on 0414
              'ego_baseline': 5, # if None, run oracle. If >=0, choose the corresponding baseline setup (note: baseline 4 = oracle with ttc_break_tie=random)
              'ttc_break_tie': 'id',
              'agent_baseline': 5,
              'stalemate_breaker': True} # 'agg_level=0' or 'b1'
# the dimension of an agent across all ts
env_configs['num_ts_in_state'] = env_configs['num_history_states'] + env_configs['num_future_states'] + 1
env_configs['agent_total_state_dim'] = env_configs['agent_state_dim'] * env_configs['num_ts_in_state']
# training configs
train_configs = {# model
                 'model': 'TwinDDQN', # [TwinDDQN, DDQN] # TODO check TwinDDQN Version!!!!
                 'gamma': env_configs['gamma'],
                 'target_update': 100, # number of policy updates until target update
                 'max_buffer_size': 200000, # max size for buffer that saves state-action-reward transitions
                 'batch_size': 128,
                 'lrt': 2e-5,
                 'tau': 0.2,
                 'exploration_timesteps': 0,
                 'value_net': 'set_transformer', # [vanilla, deep_set, deeper_deep_set, set_transformer, social_attention]
                 'future_state_dropout': 0.7, # probability of dropping out future states during training
                 'use_lstm': True, # if true, will use LSTMValueNet along with the configured value_net
                 # deep set
                 'pooling': 'mean', # ['mean', 'max']
                 # set transformer
                 'layer_norm' : True, # whether to use layer_norm in set transformer
                 'layers' : 2,
                 # 'train_every_timesteps': 4,
                 # epsilon decay for epsilon greedy
                 'eps_start': 1.0,
                 'eps_end': 0.01,
                 'eps_decay': 500,
                 # training 
                 'reward_threshold': 1000, # set impossible
                 'max_timesteps': 200000,
                 'train_every_episodes': 1, # if -1: train at every timestep. if n >= 1: train after every n timesteps
                 'save_every_timesteps': 5000,
                 'eval_every_episodes': 50,
                 'log_every_timesteps': 100,
                 'record_every_episodes': 30,
                 'seed': 0,
                 'grad_clamp': False,
                 'moving_avg_window': 100,
                 'replay_collision_episode': 0, # if 0, start every new episode fresh
                 'replay_episode': 0, # if 0, start every new episode fresh
                 'collision_episode_ratio': 0.25, # the ratio of saved collision episodes used in both training and testing. if set to 0, then don't use saved episodes
                 'interaction_episode_ratio': 0.5, # the ratio of saved interaction episodes used in both training and testing. if set to 0, then don't use interaction episodes
                 'buffer_agent_states': False, # if True, include agent-centric states in replay buffer
                 # env
                 'agent_state_dim': env_configs['agent_state_dim'],
                 'agent_total_state_dim': env_configs['agent_total_state_dim'], # state_dim of each agent
                 'max_num_other_agents_in_range': env_configs['max_num_other_agents_in_range'],
                 'state_dim': env_configs['agent_total_state_dim'] * (env_configs['max_num_other_agents_in_range']+1), # total state_dim
                 'num_ts_in_state': env_configs['num_ts_in_state'],
                 'num_history_states': env_configs['num_history_states'],
                 'num_future_states': env_configs['num_future_states'],
                 'action_dim': 2,
                 'env_action_dim': 1,
                 'max_episode_timesteps': ni_ep_len_dict[ni]}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
env = gym.make('Neighborhood-v4')
env.set_env_config(env_config)
assert env_config['max_num_other_agents'] == 25
env.set_train_config_and_device(train_configs, device)
w,b = env_config['ego_velocity_coeff']
env.log_name_ = f'out/{date}_collision-set_train_ego-vel-coeff={w},{b}_ni={ni}_na={na}.log'
dataset_path = f'../policy_network/neighborhood_v4_ddqn/collision_sets/{date}_collision-set_train_ego-vel-coeff={w},{b}_ni={ni}_na={na}'

N = 200
action = 1
episode = 0
prev_ts = 0
ts = 0
done = False

collision = 0
timeout = 0
last_save_collision = 0
collision_ep_agents_init = []

longest_episode = 0

time_start = time.time()

while collision < 200: # use this when generating dataset
    episode += 1
    total_reward = 0
    prev_ts = ts
    episode_ts = 0
    done = False

    print(f"[Episode {episode}]...")
    log(env.log_name_, f"[Episode {episode}]...")

    state = None 
    while state is None:
        state = env.reset()
    ep_agents_init = copy.deepcopy(env.agents_)

    while done == False and episode_ts < ni_ep_len_dict[env_config['ego_num_intersections_in_path']]:
        ts += 1
        episode_ts += 1
        log(env.log_name_, f"[Step {episode_ts}]...")
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done or episode_ts >= ni_ep_len_dict[env_config['ego_num_intersections_in_path']]:
            dt = (int)(time.time() - time_start)
            print(f'total reward={total_reward} | length={ts - prev_ts} | dist_driven: {info[2]} | Time: {dt//3600:02}:{dt%3600//60:02}:{dt%60}')
            longest_episode = max(longest_episode, episode_ts)
            if done:
                if info[1] == True:
                    collision += 1
                    print(f'collision={collision}')
                    collision_ep_agents_init.append(ep_agents_init)
            else:
                timeout += 1
                print(f'timeout={timeout}')

    if (collision > 0) and (collision - last_save_collision > 1):
        print(f'saving collision_ep_agents_init at collision={collision}')
        save_obj({'env_config': env_config,
                  'agents_init': collision_ep_agents_init}, dataset_path)
        last_save_collision = collision

env.close()