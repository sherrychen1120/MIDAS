import logging
import pdb
import glob
import math
import time
import pickle
from datetime import datetime
import io, sys, os, copy
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
from neighborhood_v4_ddqn.eval import eval

# === Set up environment ===
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# === Loading function ===
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# === Hyperparameters ===
# env configs
ni_ep_len_dict = {1: 200, 2: 200, 3: 200}
# experiment_date = '0414'
ni = 1
RL1_model_path = ''

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
                 'value_net': 'set_transformer', # [vanilla, deep_set, set_transformer, social_attention]
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

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-code', type=str, default='cXX')
parser.add_argument('-date', type=str, default='0000')
opt = parser.parse_args()
train_configs['seed'] = opt.seed
experiment_date = opt.date

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
torch.manual_seed(train_configs['seed'])
np.random.seed(train_configs['seed'])

model_dir = './checkpoints'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
plot_dir = './learning_curves'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
log_dir = './logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

experiment_name = f'{experiment_date}_neighborhoodv4_ddqn_{opt.code}_seed={opt.seed}'
RL2_model_path = f'./checkpoints/{experiment_name}_most-recent.pt'
log_name = f'{log_dir}/{experiment_name}.log'
if os.path.exists(log_name):
    os.remove(log_name)

log(log_name, str(env_configs))
log(log_name, str(train_configs))

# === Training Prep ===
env = gym.make('Neighborhood-v4')
env.set_env_config(env_configs)
env.set_train_config_and_device(train_configs, device)
env.log_name_ = log_name
agent = TwinDDQNAgent(train_configs, device, log_name)

## for plot
# train
scores = []
moving_averages = []
avg_velocities = []
avg_velocities_moving_averages = []
episode_lengths = []
episode_lengths_moving_averages = []
train_collisions = []
train_collision_pcts_moving_averages = []
train_timeouts = []
train_timeout_pcts_moving_averages = []
# eval
eval_scores = []
collision_pcts = []
eval_timeout_pcts = []
eval_avg_velocities = []
eval_km_per_collision = []
running_score = -float('Inf')
total_timesteps = 0
prev_save_at_timestep = 0
prev_eval_at_timestep = 0
episode_timesteps = 0
episode_collided = False
episode_replay_counter = 0
reduced_lrt = False # only reduce lrt once
prev_highest_running_score = -float('Inf')
# format for eval pfmc: [eval_score, collision_pct]
prev_best_eval_pfmc_1 = []
prev_best_eval_pfmc_2 = []
prev_best_eval_pfmc_3 = []
prev_best_eval_pfmc_4 = []
episode = 1

# load all collision episode agent inits
# TODO: make sure the env_config is the same as our env_config!
w,b = env_configs['ego_velocity_coeff']
collision_set_path_list = [f'collision_sets/0321_collision-set_train_ego-vel-coeff={w},{b}_ni=1_na=5',
                           f'collision_sets/0321_collision-set_train_ego-vel-coeff={w},{b}_ni=1_na=10',
                           f'collision_sets/0321_collision-set_train_ego-vel-coeff={w},{b}_ni=1_na=15',
                           f'collision_sets/0321_collision-set_train_ego-vel-coeff={w},{b}_ni=1_na=20',
                           f'collision_sets/0321_collision-set_train_ego-vel-coeff={w},{b}_ni=1_na=25']
interaction_set_path = f'train_sets/0601_tr5_training_set_interaction_ego-vel-coeff={w},{b}'
eval_set_path = f'eval_sets/0601_tr5_eval_set_ego-vel-coeff={w},{b}'
collision_ep_agents_init = []
for collision_set_path in collision_set_path_list:
    curr_collision_set = load_obj(collision_set_path)
    if isinstance(curr_collision_set, dict):
        saved_env_config = curr_collision_set['env_config']
        curr_collision_set = curr_collision_set['agents_init']
        log(log_name, 'saved env_config: ' + str(saved_env_config))
        for key in ['agent_stochastic_stop', 
                    'agent_shuffle_ids', 'ego_num_intersections_in_path',
                    'expanded_lane_set_depth', 'c1', 'ego_expand_path_depth', 
                    'single_intersection_type','ego_velocity_coeff','agent_velocity_coeff']: # we don't check for single_intersection_type cuz it's not gonna be the same
            if ((key not in env_configs) and (key in saved_env_config)) or \
            ((key in env_configs) and (key not in saved_env_config)) or \
            (env_configs[key] != saved_env_config[key]):
                raise Exception(f'env_config key mismatch: {key}')
    collision_ep_agents_init.append(curr_collision_set)

interaction_set = load_obj(interaction_set_path) # it's a list
eval_set = load_obj(eval_set_path) # it's a list

# CHANGE ALL THOSE IF YOU WANT TO LOAD CHECKPOINT
# Example: 
# load_model = True
# checkpoint_path = './ppo_checkpoint_episode=3000_ts=10000.pt'
# total_timesteps = 10000
# episode = 3000
load_model = False
checkpoint_path = './checkpoints/XX.pt'

if load_model:
    total_timesteps = 0
    episode = 0
    prev_save_at_timestep = total_timesteps
    prev_eval_at_timestep = total_timesteps

    has_problems = False
    log(log_name, f"Loading model from {checkpoint_path}")
    print(f"Loading model from {checkpoint_path}")
    # we will treat everything as new - just the value_net itself is from previous checkpoint
    checkpoint = agent.load(checkpoint_path)
    checkpoint_env_configs = checkpoint['env_configs']
    
    for key in env_configs.keys():
        if key in ['rl_agent_configs', 'num_other_agents']:
            continue
        if key not in checkpoint_env_configs.keys():
            print(f"key {key} not in checkpoint_env_configs. current: {key}:{env_configs[key]}")
            has_problems = True
            continue
        if (checkpoint_env_configs[key] != env_configs[key]):
            print(f"mismatch in env_config: {key} : saved: {checkpoint_env_configs[key]}, current: {env_configs[key]}")
            has_problems = True
            continue
    
    checkpoint_train_configs = checkpoint['train_configs']
    for key in train_configs.keys():
        if key in ['state_dim','seed']: # TODO
            continue
        if key not in checkpoint_train_configs.keys():
            print(f"key {key} not in checkpoint_train_configs. current: {key}:{train_configs[key]}")
            has_problems = True
            continue
        if (checkpoint_train_configs[key] != train_configs[key]):
            print(f"mismatch in train_configs: {key} : saved: {checkpoint_train_configs[key]}, current: {train_configs[key]}")
            has_problems = True
            continue

    if has_problems:
        raise Exception("Problem in loading model")

    scores = checkpoint['scores']
    eval_scores = checkpoint['eval_scores']
    collision_pcts = checkpoint['collision_pcts']
    avg_velocities = checkpoint['avg_velocities']
    episode_lengths = checkpoint['episode_lengths']
    train_collisions = checkpoint['train_collisions']
    eval_timeout_pcts = checkpoint['eval_timeout_pcts']
    train_timeouts = checkpoint['train_timeouts']
    eval_km_per_collision = checkpoint['eval_km_per_collision']

    running_score = moving_average(scores, train_configs['moving_avg_window'])

    agent.save(checkpoint, RL2_model_path)

else:
    # save init of RL2_model
    agent.save([], RL2_model_path)

# pdb.set_trace()

# timing
time_start = time.time()

def plot_curves():
    # plot
    plt.plot(np.arange(len(moving_averages)) + train_configs['moving_avg_window'], moving_averages, color='b', label='mov avg scores')
    plt.plot(np.arange(len(train_collision_pcts_moving_averages)) + train_configs['moving_avg_window'], train_collision_pcts_moving_averages, color='r', label='mov avg collision pct')
    plt.plot(np.arange(len(train_timeout_pcts_moving_averages)) + train_configs['moving_avg_window'], train_timeout_pcts_moving_averages, color='m', label='mov avg timeout pct')
    plt.title(experiment_name + ' ts=' + str(total_timesteps) + ' training curve')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(
        f"{plot_dir}/{experiment_name}_train.png",
        dpi=400,
    )
    plt.close()

    if train_configs['eval_every_episodes'] > 0:
        plt.plot(train_configs['moving_avg_window'] + np.arange(1, len(eval_scores)+1) * train_configs['eval_every_episodes'], eval_scores, color='b', label='eval score')
        plt.plot(train_configs['moving_avg_window'] + np.arange(1, len(collision_pcts)+1) * train_configs['eval_every_episodes'], collision_pcts, color='r', label='collision rate')
        plt.plot(train_configs['moving_avg_window'] + np.arange(1, len(collision_pcts)+1) * train_configs['eval_every_episodes'], eval_timeout_pcts, color='m', label='timeout rate')
        plt.plot(train_configs['moving_avg_window'] + np.arange(1, len(collision_pcts)+1) * train_configs['eval_every_episodes'], eval_km_per_collision, color='g', label='km per collision')
        
        plt.title(experiment_name + ' ts=' + str(total_timesteps) + 'eval curve')
        plt.xlabel('Episodes')
        plt.ylabel('Eval Scores / Collision Rate (%) / Km per Collision')
        plt.legend()
        plt.savefig(
            f"{plot_dir}/{experiment_name}_eval.png",
            dpi=400,
        )
        plt.close()

        plt.plot(np.arange(len(moving_averages)) + train_configs['moving_avg_window'], avg_velocities_moving_averages, color='b', label='train mov avg')
        plt.plot(train_configs['moving_avg_window'] + np.arange(1, len(eval_avg_velocities)+1) * train_configs['eval_every_episodes'], eval_avg_velocities, color='g', label='eval')
        plt.title(experiment_name + ' ts=' + str(total_timesteps) + 'Avg ego vel')
        plt.xlabel('Episodes')
        plt.ylabel('Avg ego velocity')
        plt.legend()
        plt.savefig(
            f"{plot_dir}/{experiment_name}_avg_velocities.png",
            dpi=400,
        )
        plt.close()

        plt.plot(np.arange(len(moving_averages)) + train_configs['moving_avg_window'], episode_lengths_moving_averages, color='g')
        plt.title(experiment_name + ' ts=' + str(total_timesteps) + 'ep lengths (mov avg)')
        plt.xlabel('Episodes')
        plt.ylabel('Episode lengths')
        plt.legend()
        plt.savefig(
            f"{plot_dir}/{experiment_name}_ep_lengths.png",
            dpi=400,
        )
        plt.close()

def change_to_self_play(loaded_agents):
    updated_loaded_agents = copy.deepcopy(loaded_agents)
    for agent_id, agent in updated_loaded_agents.items():
        if int(agent_id) == 0:
            continue
        # this essentially changes this agent to an rl agent
        agent.rl_model_path_ = RL2_model_path 
        agent.train_config_ = train_configs
        agent.device_ = device
    return updated_loaded_agents

def append_action_and_reward(action, reward, hist_actions, hist_rewards):
    num_a_r_saved = env_configs['num_future_states'] * env_configs['time_gap']+1
    if len(hist_actions) == num_a_r_saved:
        hist_actions = hist_actions[1:]
        hist_rewards = hist_rewards[1:]
    hist_actions.append(action)
    hist_rewards.append(reward)
    return hist_actions, hist_rewards


print("=== Training ===")

while total_timesteps < train_configs['max_timesteps']:
    log(log_name, f"[Episode {episode}] Starting...")
    # check replay config
    if train_configs['replay_collision_episode'] > 0 and train_configs['replay_episode'] > 0:
        raise Exception('Only one of replay_collision_episode and replay_episode can be > 0!')
    # decide whether to use a saved collision episode or to use a randomly generated episode
    ep_sample = np.random.random()
    if (ep_sample < train_configs['collision_episode_ratio']): # use collision_ep_agents_init
        # pick a set
        random_set_idx = np.random.randint(len(collision_ep_agents_init))
        curr_set = collision_ep_agents_init[random_set_idx]
        # pick an episode from that set
        random_ep_idx = np.random.randint(len(curr_set))
        env.env_config_['num_other_agents'] = len(curr_set[random_ep_idx])-1
        log(log_name, f"[Episode {episode}] Using saved collision episode {random_ep_idx} from {collision_set_path_list[random_set_idx]}")
        
        loaded_agents = curr_set[random_ep_idx]
        if 'all_selfplay' in env_configs.keys() and env_configs['all_selfplay'] == True:
            updated_loaded_agents = change_to_self_play(loaded_agents)
            parametric_state, ego_b1 = env.reset(use_saved_agents=updated_loaded_agents)
        else:
            parametric_state, ego_b1 = env.reset(use_saved_agents=loaded_agents)

    elif (ep_sample < train_configs['collision_episode_ratio'] + train_configs['interaction_episode_ratio']): # use interaction_set
        # pick an episode from the set
        random_ep_idx = np.random.randint(len(interaction_set))
        # interaction set is saved as a list of dicts, each with its own env_config
        saved_env_config = interaction_set[random_ep_idx]['env_config']
        loaded_agents = interaction_set[random_ep_idx]['agents_init']
        
        # the saved env config for each episode should have these fields coincide with the env config during eval
        for key in ['agent_stochastic_stop', 
                    'agent_shuffle_ids', 'ego_num_intersections_in_path',
                    'expanded_lane_set_depth', 'c1', 'ego_expand_path_depth', 
                    'single_intersection_type','ego_velocity_coeff','agent_velocity_coeff']: # 'ego_expand_path_depth' and 'c1' not tested for 0313 and 0314 eval sets
            if ((key not in env.env_config_) and (key in saved_env_config)) or \
            ((key in env.env_config_) and (key not in saved_env_config)) or \
            (env.env_config_[key] != saved_env_config[key]):
                raise Exception(f'env_config key mismatch {key}. saved: {saved_env_config[key]}. env: {env.env_config_[key]}')

        # only change num_other_agents!
        env.env_config_['num_other_agents'] = saved_env_config['num_other_agents']

        if 'all_selfplay' in env_configs.keys() and env_configs['all_selfplay'] == True:
            updated_loaded_agents = change_to_self_play(loaded_agents)
            parametric_state, ego_b1 = env.reset(use_saved_agents=updated_loaded_agents)
        else:
            parametric_state, ego_b1 = env.reset(use_saved_agents=loaded_agents)

        log(log_name, f"[Episode {episode}] Using saved interaction episode {random_ep_idx}")
    else: # randomly generate or consult replay configs
        # reset depending on replay config
        if train_configs['replay_collision_episode'] > 0:
            if ((episode_replay_counter == 0) and (episode_collided == True)) or \
            ((episode_replay_counter > 0) and (episode_replay_counter < train_configs['replay_collision_episode'])): 
                episode_replay_counter += 1
                parametric_state, ego_b1 = env.reset(use_prev_episode=True)
                log(log_name, f"[Episode {episode}] Using previous episode for the {episode_replay_counter}-th time")
                log(log_name, str(parametric_state))
            else:
                episode_replay_counter = 0
                parametric_state, ego_b1 = env.reset()
        elif train_configs['replay_episode'] > 0:
            if (episode > 1 and episode_replay_counter < train_configs['replay_episode']):
                episode_replay_counter += 1
                parametric_state, ego_b1 = env.reset(use_prev_episode=True)
                log(log_name, f"[Episode {episode}] Using previous episode for the {episode_replay_counter}-th time")
                log(log_name, str(parametric_state))
            else:
                episode_replay_counter = 0
                parametric_state, ego_b1 = env.reset()
        else:
            # start fresh with random na
            num_other_agents = np.random.randint(env_configs['max_num_other_agents']+1)
            env_configs['num_other_agents'] = num_other_agents
            log(log_name, f"[Episode {episode}] Start fresh. num_other_agents={num_other_agents}")
            
            if 'all_selfplay' in env_configs.keys() and env_configs['all_selfplay'] == True:
                if os.path.exists(RL2_model_path):
                    rl_agent_configs = [(RL2_model_path,1.0)]
                    env_configs['rl_agent_configs'] = rl_agent_configs
                    log(log_name, "Updated env_config: " + str(env_configs))
                else:
                    raise Exception(f"{RL2_model_path} doesn't exist")

            env.set_env_config(env_configs)
            parametric_state, ego_b1 = env.reset()
    
    hist_parametric_state = parametric_state
    hist_actions = []
    hist_rewards = []
    hist_agent_states,_,_,_,_ = env.generate_agent_states()

    episode_timesteps = 0
    score = 0
    episode_collided = False
    done = False

    while (done == False) and (episode_timesteps < train_configs['max_episode_timesteps']):
        total_timesteps += 1
        episode_timesteps += 1 

        # select action
        if env_configs['num_future_states'] > 0:
            parametric_state_till_now = truncate_state_till_now(parametric_state, env_configs)
        else:
            parametric_state_till_now = parametric_state
        parametric_state_ts = torch.from_numpy(parametric_state_till_now).unsqueeze(0).float().to(device)
        ## for printing
        action = int(agent.select_action(parametric_state_ts, ego_b1, total_timesteps))
        # take action in env:
        next_state, reward, done, info = env.step(action) # step processed action
        action_str = "selected action: " + str(action)
        log(log_name, f'[Episode {episode} ts={episode_timesteps}] reward={reward:.1f} | {action_str}')
        
        # process next state
        parametric_state = next_state
        
        # save
        hist_actions, hist_rewards = append_action_and_reward(action, reward, hist_actions, hist_rewards)
        if env_configs['num_future_states'] == 0:
            agent.replay_buffer.add(hist_parametric_state, parametric_state, 
                                    hist_actions[0], hist_rewards[0], float(done), info[4]) # info[4] is ego_b1
        else:
            if len(hist_actions) == (env_configs['num_future_states'] * env_configs['time_gap'] + 1):
                agent.replay_buffer.add(hist_parametric_state, parametric_state, 
                                        hist_actions[0], hist_rewards[0], 0.0, info[4]) # info[4] is ego_b1
        hist_parametric_state = parametric_state
        
        # if buffer_agent_states
        if train_configs['buffer_agent_states']:
            agent_states, agent_rewards, agent_actions, agent_dones, agent_b1s = env.generate_agent_states()
            for i in range(len(agent_states)):
                agent.replay_buffer.add(hist_agent_states[i], agent_states[i], agent_actions[i], agent_rewards[i], float(agent_dones[i]), agent_b1s[i])
            hist_agent_states = agent_states

        # update agent if train_every_episodes == -1 (train at every ts)
        if train_configs['train_every_episodes'] == -1:
            if (total_timesteps >= train_configs['exploration_timesteps']) and \
                (agent.replay_buffer.size() > train_configs['batch_size']):
                loss = agent.train()
                agent.save(scores, RL2_model_path, train_configs=train_configs, env_configs=env_configs, \
                            eval_scores=eval_scores, collision_pcts=collision_pcts, \
                            episode_lengths=episode_lengths, avg_velocities=avg_velocities,\
                            train_collisions=train_collisions, eval_timeout_pcts=eval_timeout_pcts, \
                            train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)
                if ((total_timesteps + 1) % train_configs['log_every_timesteps'] == 0):
                    log(log_name, f'[Total timesteps {total_timesteps}] At every timestamp, Update and save agent to {RL2_model_path}')
            
        score += reward

        if (done == True):
            episode_collided = info[1]

            # when an episode ends, repeat the last state t_future times until 1 entry with done=1 is saved
            saved_done = 0.0
            num_repeat_steps = env_configs['num_future_states'] * env_configs['time_gap']
            for t in range(1, num_repeat_steps+1):
                # take action in env:
                next_state, reward, done, info = env.step(0)
                log(log_name, f'[Episode {episode} ts={episode_timesteps}] repeat {t}/{num_repeat_steps} after done. reward={reward:.1f}')
                
                # process next state
                parametric_state = next_state
                # save
                hist_actions, hist_rewards = append_action_and_reward(0, reward, hist_actions, hist_rewards)
                if t == num_repeat_steps:
                    saved_done = 1.0
                agent.replay_buffer.add(hist_parametric_state, parametric_state, 
                                        hist_actions[0], hist_rewards[0], saved_done, info[4]) # info[4] is ego_b1
                hist_parametric_state = parametric_state

    episode += 1
    scores.append(score)
    avg_velocities.append(info[0])
    if (info[1] == True):
        train_collisions.append(100)
    else:
        train_collisions.append(0)
    if (done == False):
        train_timeouts.append(100)
    else:
        train_timeouts.append(0)
    episode_lengths.append(episode_timesteps)
    running_score = moving_average(scores, train_configs['moving_avg_window'])

    # update agent if train_every_episodes > 0 (train after complete episodes)
    if train_configs['train_every_episodes'] > 0 and (episode % train_configs['train_every_episodes'] == 0):
        if (total_timesteps >= train_configs['exploration_timesteps']) and \
            (agent.replay_buffer.size() > train_configs['batch_size']):
            num_train_iterations = sum(episode_lengths[(-train_configs['train_every_episodes']):])
            for i in range(1, num_train_iterations+1):
                loss = agent.train()
                dt = (int)(time.time() - time_start)
                if i % 10 == 0:
                    log(log_name, f'[Episode {episode} | Total timesteps = {total_timesteps}] Updating agent {i} / {num_train_iterations} | Time: {dt//3600:02}:{dt%3600//60:02}:{dt%60:02}')
            log(log_name, f'[Episode {episode} | Total timesteps = {total_timesteps}] Saving most recent model to {RL2_model_path} ...')
            agent.save(scores, RL2_model_path, train_configs=train_configs, env_configs=env_configs, \
                            eval_scores=eval_scores, collision_pcts=collision_pcts, \
                            episode_lengths=episode_lengths, avg_velocities=avg_velocities,\
                            train_collisions=train_collisions, eval_timeout_pcts=eval_timeout_pcts, \
                            train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)
            
    
    # append running score and save highest running score model
    if (total_timesteps >= train_configs['exploration_timesteps']) and (len(scores) > train_configs['moving_avg_window']):
        # trackers of moving averages
        moving_averages.append(running_score)
        avg_velocities_moving_averages.append(moving_average(avg_velocities, train_configs['moving_avg_window']))
        episode_lengths_moving_averages.append(moving_average(episode_lengths, train_configs['moving_avg_window']))
        train_collision_pcts_moving_averages.append(moving_average(train_collisions, train_configs['moving_avg_window']))
        train_timeout_pcts_moving_averages.append(moving_average(train_timeouts, train_configs['moving_avg_window']))

        # Adjust learning rate
        if (len(train_collision_pcts_moving_averages) > 50):
            max_mov_avg_collision_pct = max(train_collision_pcts_moving_averages[-50:])
            if (max_mov_avg_collision_pct < 10) and (reduced_lrt == False):
                target_lrt = 1e-5 # keep min at 1e-5
                log(log_name, f'[Total timesteps {total_timesteps}] Reducing lrt with max last 50 mov avg collision: {max_mov_avg_collision_pct}. Reducing to {target_lrt}')
                agent.reduce_lrt(target_lrt)
                reduced_lrt = True

        # Eval model at every several ts
        if train_configs['eval_every_episodes'] > 0:
            if (episode % train_configs['eval_every_episodes'] == 0):
                log(log_name, f'[Total timesteps {total_timesteps}] Evaluating model ...')
                agent.value_net.eval()
                # == Eval ==
                # save current random state and set np random state to a new one
                log(log_name, 'generating eval seed...')
                eval_seed = np.random.randint(65536)
                log(log_name, str(eval_seed))
                while (eval_seed == train_configs['seed']):
                    eval_seed = np.random.randint(65536)
                    log(log_name, str(eval_seed))
                log(log_name, f'eval seed: {eval_seed}')
                train_random_state = np.random.get_state()
                np.random.seed(eval_seed)
                # create eval env
                eval_env = gym.make('Neighborhood-v4')
                copy_eval_env_configs = copy.deepcopy(env_configs) # because eval will change env_configs, we pass in a copy
                eval_env.set_env_config(copy_eval_env_configs)
                eval_env.log_name_ = log_name
                eval_env.max_v_value_ = max_v_value
                eval_env.max_rel_v_value_ = max_rel_v_value
                # call eval
                eval_score, collision_pct, success_pct, eval_avg_velocity, km_per_collision = eval(agent, eval_env, train_configs, env_configs, [], device, log_name, collision_ep_agents_init=eval_set, during_training=True)
                timeout_pct = 100 - success_pct - collision_pct
                log(log_name, f'[Eval] avg score = {eval_score:.2f} | collision = {collision_pct}% | success = {success_pct}% | timeout = {timeout_pct}% | km_per_collision = {km_per_collision} | avg_velocity = {eval_avg_velocity}')
                # set np random state back to the train seed
                np.random.set_state(train_random_state)
                agent.value_net.train()

                # == Eval trackers ==
                eval_scores.append(eval_score)
                collision_pcts.append(collision_pct)
                eval_timeout_pcts.append(timeout_pct)
                eval_avg_velocities.append(eval_avg_velocity)
                eval_km_per_collision.append(km_per_collision)

                # == Eval saving ==
                # eval_pfmc_1
                if (len(prev_best_eval_pfmc_1) == 0) or \
                    ((eval_score > prev_best_eval_pfmc_1[0]) and (collision_pct <= prev_best_eval_pfmc_1[1])):
                    prev_best_eval_pfmc_1 = [eval_score, collision_pct]
                    log(log_name, f"Best eval pfmc 1: [{eval_score:.2f}, {collision_pct:.1f}%] at {timestring}")
                    filename = f'{model_dir}/{experiment_name}_best-eval-pfmc-1.pt'
                    agent.save(scores, filename, train_configs=train_configs, env_configs=env_configs, \
                                eval_scores=eval_scores, collision_pcts=collision_pcts, \
                                episode_lengths=episode_lengths, avg_velocities=avg_velocities,\
                                train_collisions=train_collisions, eval_timeout_pcts=eval_timeout_pcts, \
                                train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)
                # eval_pfmc_2
                if (len(prev_best_eval_pfmc_2) == 0) or \
                    (eval_score - collision_pct > prev_best_eval_pfmc_2[0] - prev_best_eval_pfmc_2[1]):
                    prev_best_eval_pfmc_2 = [eval_score, collision_pct]
                    log(log_name, f"Best eval pfmc 2: [{eval_score:.2f}, {collision_pct:.1f}%] with difference {eval_score - collision_pct} at {timestring}")
                    filename = f'{model_dir}/{experiment_name}_best-eval-pfmc-2.pt'
                    agent.save(scores, filename, train_configs=train_configs, env_configs=env_configs, \
                                eval_scores=eval_scores, collision_pcts=collision_pcts, \
                                episode_lengths=episode_lengths, avg_velocities=avg_velocities,\
                                train_collisions=train_collisions, eval_timeout_pcts=eval_timeout_pcts, \
                                train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)
                # eval_pfmc_3
                if (len(prev_best_eval_pfmc_3) == 0) or \
                    (eval_score > prev_best_eval_pfmc_3[0]):
                    prev_best_eval_pfmc_3 = [eval_score, collision_pct]
                    log(log_name, f"Best eval pfmc 3: [{eval_score:.2f}, {collision_pct:.1f}%] at {timestring}")
                    filename = f'{model_dir}/{experiment_name}_best-eval-pfmc-3.pt'
                    agent.save(scores, filename, train_configs=train_configs, env_configs=env_configs, \
                                eval_scores=eval_scores, collision_pcts=collision_pcts, \
                                episode_lengths=episode_lengths, avg_velocities=avg_velocities,\
                                train_collisions=train_collisions, eval_timeout_pcts=eval_timeout_pcts, \
                                train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)
                
                # eval_pfmc_4
                if (len(prev_best_eval_pfmc_4) == 0) or \
                    (success_pct > prev_best_eval_pfmc_4[1]):
                    prev_best_eval_pfmc_4 = [eval_score, success_pct]
                    log(log_name, f"Best eval pfmc 4: [{eval_score:.2f}, success_pct={success_pct}] at {timestring}")
                    filename = f'{model_dir}/{experiment_name}_best-eval-pfmc-4.pt'
                    agent.save(scores, filename, train_configs=train_configs, env_configs=env_configs, \
                                eval_scores=eval_scores, collision_pcts=collision_pcts, \
                                episode_lengths=episode_lengths, avg_velocities=avg_velocities,\
                                train_collisions=train_collisions, eval_timeout_pcts=eval_timeout_pcts, \
                                train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)

        # save learning curves
        if (episode % train_configs['record_every_episodes'] == 0):
            plot_curves()

        # if avg reward > 1000 then save and stop training:
        if (running_score > prev_highest_running_score):
            # logger.info("##### Success! #####")
            timestring = (datetime.now()).strftime("%m%d-%H:%M")
            log(log_name, f"Highest training running score {running_score} at {timestring}")
            filename = f'{model_dir}/{experiment_name}_highest-running-score.pt'
            agent.save(scores, filename, train_configs=train_configs, env_configs=env_configs, \
                        eval_scores=eval_scores, collision_pcts=collision_pcts, \
                        episode_lengths=episode_lengths, avg_velocities=avg_velocities,\
                        train_collisions=train_collisions, eval_timeout_pcts=eval_timeout_pcts, \
                        train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)
            prev_highest_running_score = running_score

            if running_score >= train_configs['reward_threshold']:
                log(log_name, f"##### Success! ##### running_score={running_score}")
                break
        
    dt = (int)(time.time() - time_start)
    log(log_name, "[Episode {}] [Total timesteps {}] length: {} | score: {:.1f} | Avg. last {} scores: {:.3f} | Time: {:02}:{:02}:{:02}\n"\
          .format(episode, total_timesteps, episode_timesteps, score, train_configs['moving_avg_window'], running_score, dt//3600, dt%3600//60, dt%60))

    # Save episode
    if total_timesteps > train_configs['exploration_timesteps'] and \
        (total_timesteps - prev_save_at_timestep >= train_configs['save_every_timesteps']):
        log(log_name, f'[Total timesteps {total_timesteps}] Saving model ...')
        timestring = (datetime.now()).strftime("%m%d-%H:%M")
        filename = f'{model_dir}/{experiment_name}_{timestring}_ts={total_timesteps}_running-score={running_score}.pt'
        agent.save(scores, filename, train_configs=train_configs, env_configs=env_configs, \
                    eval_scores=eval_scores, collision_pcts=collision_pcts, \
                    episode_lengths=episode_lengths, avg_velocities=avg_velocities, \
                    train_collisions=train_collisions, eval_timeout_pcts=eval_timeout_pcts, \
                    train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)
        prev_save_at_timestep += train_configs['save_every_timesteps']
    
# Finishing training
log(log_name, f'[Episode {episode}][Total timesteps {total_timesteps}] Done Training. Saving final model')
filename = f'{model_dir}/{experiment_name}_ep={episode}_final_ts={total_timesteps}.pt'
agent.save(scores, filename, train_configs=train_configs, env_configs=env_configs, \
            eval_scores=eval_scores, collision_pcts=collision_pcts, \
            episode_lengths=episode_lengths, avg_velocities=avg_velocities, train_collisions=train_collisions, \
            eval_timeout_pcts=eval_timeout_pcts, train_timeouts=train_timeouts, eval_km_per_collision=eval_km_per_collision)

# plot
plot_curves()
