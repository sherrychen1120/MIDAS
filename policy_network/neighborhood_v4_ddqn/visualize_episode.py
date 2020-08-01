import logging
import pdb, copy
import glob
import math
import argparse
import time
from datetime import datetime
import io, sys, os, pickle
import base64
import argparse
import shutil

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
from gym_road_interactions.viz_utils import write_nonsequential_idx_video

from neighborhood_v4_ddqn.models import *
from neighborhood_v4_ddqn.utils import *

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# === Set up environment ===
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# make this part independent of the checkpoint seeds
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
eval_seed = np.random.randint(65536)
torch.manual_seed(eval_seed)
np.random.seed(eval_seed)

def model_test_run(agent, env, ep_ids, set_stats,
                   train_configs, env_configs, device, log_name, model_viz_path,
                   curr_ego_b1 = None, # when set to None, use the ego_b1 provided in the set_stats. Otherwise, use provided curr_ego_b1 for the whole set
                   ego_action_noise = None # when set to None, use the ego_action_noise provided in train_configs. Otherwise, use provided ego_action_noise for the whole set
                   ): 
    ni_ep_len_dict = {1: 200, 2:200, 3:200}

    total_reward = 0
    total_dist_driven = 0
    # total_normalized_brake = 0
    total_time_to_finish = 0
    collision_cnt = 0
    success_cnt = 0 # success means ego reaches the end before timeout

    episode = 0
    episode_score = 0
    episode_length = 0

    num_episodes = len(ep_ids)
    log(log_name, f'number of test run episodes: {num_episodes}')

    # reset ego_action_noise if not None
    if ego_action_noise is not None:
        print(f"Reset env ego_action_noise to {ego_action_noise}")
        env.env_config_['ego_action_noise'] = ego_action_noise

    for ep_id in ep_ids:
        episode += 1

        episode_path = f'{model_viz_path}/{ep_id}'

        if os.path.exists(episode_path):
            shutil.rmtree(episode_path)
        os.mkdir(episode_path)
        logger.info(f'visualizing episode {ep_id} at {episode_path}')

        # else:
        # ONLY CHANGE num_other_agents in env.env_config_! The caller should instantiate the env with the correct env_config. 
        # Only num_other_agents is different across episodes
        if isinstance(set_stats, dict):
            saved_env_config = set_stats[ep_id]['env_config']
            loaded_agents = set_stats[ep_id]['agents_init']
        else:
            for ep_entry in set_stats:
                if ep_entry['ep_id'] == ep_id:
                    saved_env_config = ep_entry['env_config']
                    loaded_agents = ep_entry['agents_init']
                    break
        
        # the saved env config for each episode should have these fields coincide with the env config during eval
        # TODO 'ttc_break_tie','stalemate_breaker', 'use_default_ego'
        for key in ['agent_stochastic_stop', 
            'agent_shuffle_ids', 'single_intersection_type',
            'expanded_lane_set_depth', 'ego_velocity_coeff', 'agent_velocity_coeff', 
            'ego_expand_path_depth', 'c1']:
            if ((key not in env.env_config_) and (key in saved_env_config)) or \
            ((key in env.env_config_) and (key not in saved_env_config)) or \
            (env.env_config_[key] != saved_env_config[key]):
                raise Exception(f'eval env_config key mismatch {key}. saved: {saved_env_config[key]}. env: {env.env_config_[key]}')

        # change num_other_agents!
        # add this check to avoid the issue of incorrect save env config
        if len(loaded_agents) != (saved_env_config['num_other_agents'] + 1):
            env.env_config_['num_other_agents'] = len(loaded_agents) - 1
        else:
            env.env_config_['num_other_agents'] = saved_env_config['num_other_agents']
        # change num intersections in path and max num other agents according to the recorded episode
        print('Model ni=' + str(env.env_config_['ego_num_intersections_in_path']) + ' max_na=' + str(env.env_config_['max_num_other_agents']))
        env.env_config_['ego_num_intersections_in_path'] = saved_env_config['ego_num_intersections_in_path']
        env.env_config_['max_num_other_agents'] = saved_env_config['max_num_other_agents']
        env.env_config_['max_episode_timesteps'] = ni_ep_len_dict[saved_env_config['ego_num_intersections_in_path']]
        train_configs['max_episode_timesteps'] = ni_ep_len_dict[saved_env_config['ego_num_intersections_in_path']]
        print('Updated to saved ni=' + str(env.env_config_['ego_num_intersections_in_path']) + ' max_na=' + str(env.env_config_['max_num_other_agents']))
        parametric_state, ego_b1 = env.reset(use_saved_agents=loaded_agents)
        env.render(episode_path)
        # reset ego_b1 if set within range
        if curr_ego_b1 is not None and curr_ego_b1 <= 1 and curr_ego_b1 >= -1:
            print(f"Reset env ego_b1 to {curr_ego_b1}")
            env.reset_ego_b1(curr_ego_b1)
        
        # debug
        num_other_agents = env.env_config_['num_other_agents']
        max_num_other_agents = env.env_config_['max_num_other_agents']
        log(log_name, f"[Viz Ep {episode}] Using ep_id={ep_id}. num other agents: {num_other_agents}. max_num_other_agents: {max_num_other_agents}")
        print(f"[Viz Ep {episode}] Using ep_id={ep_id}. num other agents: {num_other_agents}. max_num_other_agents: {max_num_other_agents}")

        done = False
        episode_length = 0
        episode_score = 0

        while (done == False) and (episode_length < train_configs['max_episode_timesteps']):
            # select action
            if not env.env_config_['use_default_ego']:
                if train_configs['num_future_states'] > 0:
                    parametric_state_till_now = truncate_state_till_now(parametric_state, env_configs)
                else:
                    parametric_state_till_now = parametric_state
                parametric_state_ts = torch.from_numpy(parametric_state_till_now).unsqueeze(0).float().to(device) # 1*(state_dim)
                action = agent.select_action(parametric_state_ts, ego_b1, 0, test=True)
            else:
                action = 1 # will be overriden in env anyways
            next_state, reward, done, info = env.step(action) # step processed action
            action_str = "selected action: " + str(action)
            log(log_name, f'[Viz Ep {episode} ts={episode_length}] reward={reward:.1f} | {action_str}')
            
            parametric_state = next_state # !!!

            episode_score += reward
            episode_length += 1

            env.render(episode_path)

        if (info[1] == True):
            collision_cnt += 1
            end_status = 'collision'
        elif done == True:
            success_cnt += 1
            end_status = 'success'
        else:
            end_status = 'timeout'

        log(log_name, f"[Viz Ep {episode} len={episode_length}] episode score={episode_score:.1f} | episode_dist_driven = {info[2]} | {end_status}")
        print(f"[Viz Ep {episode} len={episode_length}] episode score={episode_score:.1f} | episode_dist_driven = {info[2]} | {end_status}")
        
        total_reward += episode_score
        total_dist_driven += info[2]
        total_time_to_finish += episode_length

        # make a video
        fps = 5
        img_wildcard = f"{episode_path}/*.png"
        output_fpath = f"{episode_path}/len={episode_length:.2f}_score={episode_score:.2f}_{end_status}.mp4"
        cmd = write_nonsequential_idx_video(img_wildcard, output_fpath, fps, True)
        
    # trackers
    avg_score = total_reward / num_episodes
    avg_time_to_finish = total_time_to_finish / float(num_episodes)
    # avg_normalized_brake = total_normalized_brake / num_episodes
    collision_pct = collision_cnt / num_episodes * 100
    success_pct = success_cnt / num_episodes * 100
    if collision_cnt == 0:
        km_per_collision = total_dist_driven / 1000.0
    else:
        km_per_collision = (total_dist_driven / 1000.0) / collision_cnt
    
    return avg_score, avg_time_to_finish, collision_pct, success_pct, km_per_collision

# ======== INPUT ========
seeds = [0]
model_date = '0717'
seed_date_hyperparam_combos = [(0, model_date)]
gen_date = '0724'
code = 'c4-1'
lrt = '2e-05'
curr_ego_b1s = [None] # [-1,-0.5,0.0,0.5,1]
ego_action_noise = None
set_stats_name = 'testing_set_stats'
model_suffix = 'best-eval-pfmc-2'
model_suffix_short = 'bep2'

# TODO: Specify the ep_ids # This specifies which episodes you want to run 
ep_ids = []

# TODO: Specify the path of the dataset where the ep_ids are from
set_stats_path = f'./datasets/testing_set'

# =====================
ni_ep_len_dict = {1: 200, 2:200, 3:200}
ni = 1
max_na = 25

filename_suffix = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default=code + "-" + model_suffix_short)
    opt = parser.parse_args()
    name = opt.name

    all_seeds_avg_scores = 0
    all_seeds_avg_time_to_finish = 0
    all_seeds_avg_collision_pct = 0
    all_seeds_avg_success_pct = 0
    all_seeds_avg_km_per_collision = 0

    time_start = time.time()

    for curr_ego_b1 in curr_ego_b1s:
        for combo in seed_date_hyperparam_combos:
            seed, date = combo
            # === Load checkpoint ===
            experiment_name = f'{date}_neighborhoodv4_ddqn_{code}'
            model_name = f'{experiment_name}_seed={seed}_{model_suffix}'
            checkpoint_path = f'./checkpoints/{model_name}.pt'
            log_name = f'./viz_logs/{gen_date}_viz_model={name}_{set_stats_name}'
            
            if curr_ego_b1 is not None:
                log_name += f'_ego-b1={curr_ego_b1}'
            if ego_action_noise is not None:
                log_name += f'_ego-action-noise={ego_action_noise}'
            if filename_suffix is not None:
                log_name += filename_suffix
            log_name += '.log'
            
            checkpoint = torch.load(checkpoint_path)
            train_configs = checkpoint['train_configs']
            env_configs = checkpoint['env_configs']
            print(checkpoint_path)
            print(train_configs)
            print(env_configs)

            # use v3 or v4 c0 checkpoints in v4 c1 or later env
            if 'num_history_states' not in env_configs.keys():
                env_configs['num_history_states'] = 0
            if 'num_future_states' not in env_configs.keys():
                env_configs['num_future_states'] = 0
                train_configs['num_future_states'] = 0
            if 'time_gap' not in env_configs.keys():
                env_configs['time_gap'] = 0
            if 'stalemate_horizon' not in env_configs.keys():
                env_configs['stalemate_horizon'] = 4 # make it larger than 1 cuz otherwise every state is considered stalemate
            if 'agent_total_state_dim' not in env_configs.keys():
                env_configs['num_ts_in_state'] = env_configs['num_history_states'] + env_configs['num_future_states'] + 1
                env_configs['agent_total_state_dim'] = env_configs['agent_state_dim'] * env_configs['num_ts_in_state']
                train_configs['agent_total_state_dim'] = env_configs['agent_total_state_dim']
                train_configs['agent_state_dim'] = env_configs['agent_state_dim']
                train_configs['state_dim'] = env_configs['agent_total_state_dim'] * (env_configs['max_num_other_agents']+1)
                train_configs['num_ts_in_state'] = env_configs['num_ts_in_state']

            print('Setting test env diff. from training... ')
            env_configs['use_default_ego'] = use_default_ego
            env_configs['stalemate_breaker'] = True
            env_configs['ttc_break_tie'] = 'id'
            print(env_configs)

            # model_viz_path
            model_viz_path = f'./visualization/{model_name}_{set_stats_name}'
            if curr_ego_b1 is not None:
                model_viz_path += f'_ego-b1={curr_ego_b1}'
            if ego_action_noise is not None:
                model_viz_path += f'_ego-action-noise={ego_action_noise}'
            
            # make folder
            if not os.path.exists(model_viz_path):
                os.mkdir(model_viz_path)
                
            # set_stats
            set_stats = load_obj(set_stats_path)

            # === Prep Env ===
            # Remember to set the env_config of env before passing it to model_test_run()!
            env = gym.make('Neighborhood-v4')
            env.set_env_config(env_configs)
            env.set_train_config_and_device(train_configs, device)
            env.log_name_ = log_name
            if not use_default_ego:
                if train_configs['model'] == 'TwinDDQN':
                    agent = TwinDDQNAgent(train_configs, device, log_name)
                else:
                    agent = DDQNAgent(train_configs, device)
                agent.value_net.eval()
                agent.load(checkpoint_path)
            else:
                agent = None

            # === eval ===
            model_test_run_results = model_test_run(agent, env, ep_ids, set_stats,
                                                    train_configs, env_configs, device, log_name, model_viz_path, 
                                                    curr_ego_b1=curr_ego_b1, ego_action_noise=ego_action_noise)
            avg_score, avg_time_to_finish, collision_pct, success_pct, km_per_collision = model_test_run_results
            
            # === tracker ===
            all_seeds_avg_scores += avg_score
            all_seeds_avg_time_to_finish += avg_time_to_finish
            all_seeds_avg_collision_pct += collision_pct
            all_seeds_avg_success_pct += success_pct
            all_seeds_avg_km_per_collision += km_per_collision

            dt = (int)(time.time() - time_start)
            print(f'[Seed {seed}] Model test run: avg score = {avg_score:.3f} | ttf = {avg_time_to_finish} | collision = {collision_pct}% | success = {success_pct}% |  | km_per_collision = {km_per_collision}')
            print("Time: {:02}:{:02}:{:02}".format(dt//3600, dt%3600//60, dt%60))

        all_seeds_avg_scores /= len(seeds)
        all_seeds_avg_time_to_finish /= len(seeds)
        all_seeds_avg_collision_pct /= len(seeds)
        all_seeds_avg_success_pct /= len(seeds)
        all_seeds_avg_km_per_collision /= len(seeds)
        
        print(f'Model test run over seeds: avg score = {all_seeds_avg_scores:.3f} | ttf = {all_seeds_avg_time_to_finish} | collision = {all_seeds_avg_collision_pct}% | success = {all_seeds_avg_success_pct}% | km_per_collision = {all_seeds_avg_km_per_collision}')
