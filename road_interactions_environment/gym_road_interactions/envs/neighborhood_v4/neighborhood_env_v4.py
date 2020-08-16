# Closed neighborhood environment with a roundabout, 4 t-intersections

# Python
import pdb, copy, os
import pickle
import numpy as np
from numpy import linalg as la
import math
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import logging
from queue import Queue
from typing import Any, Dict, Tuple, List, Set
from collections import deque
import cv2
import random
import time
# Argoverse
from argoverse.utils.se3 import SE3
# gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding
# env
from .neighborhood_env_v4_agents import *
from .neighborhood_env_v4_utils import *
# utils
from gym_road_interactions.utils import conditional_log, remap, rotation_matrix_z
from gym_road_interactions.viz_utils import visualize_agent
from gym_road_interactions.core import AgentType, Position, Agent, ObservableState, Observation, LaneSegment
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

DEFAULT_TTC = 8.0
DONE_THRES = 0.5 # distance threshold for completing the task
DT = 0.1 # 10Hz
FAD_COEFF = 0.3

# load map
lane_segments_path = '../maps/neighborhood_v0_map_lane_segments.pkl'
constants_path = '../maps/neighborhood_v0_map_constants.pkl'
intersection_id_dict = '../maps/neighborhood_v0_intersection_id_dict.pkl'
with open(lane_segments_path, 'rb') as f:
    lane_segments = pickle.load(f)
with open(constants_path, 'rb') as f:
    map_constants = pickle.load(f)
with open(intersection_id_dict, 'rb') as f:
    intersection_id_dict = pickle.load(f)

class NeighborhoodEnvV4(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self) -> None:
        self.lane_segments_ = lane_segments
        plt.ion()
        self.dt_ = DT
        self.default_ttc_ = DEFAULT_TTC
        self.done_thres_ = DONE_THRES 

        # trackers
        self.curr_ts_ = 0.0
        self.ego_vel_sum_ = 0.0 # sum up all ego velocity values to calculate average velocity
        self.log_name_ = None

        # data members
        self.agents_ = {}
        self.prev_agents_ = {} # save a copy of a set of initialization of agent state
        self.env_config_ = None
        self.train_config_ = None
        self.device_ = None

        # frame values for normalization
        self.max_xy_value_ = 2 * map_constants['a3'] + 46
        self.max_v_value_ = 6.7
        self.max_rel_v_value_ = self.max_v_value_ + 5.6
        self.fad_coeff_ = FAD_COEFF

        # TIME
        self.time_start_ = time.time()
        self.profiling_ = False

        self.ttc_dp_ = None
        self.max_retry_ = 20
        self.ttc_dps = [] # debug

        # manually defined interacting agents for interaction sets (not used)
        self.interacting_agents_ = [] 

        # episode_record
        # format: [np.array([x,y,theta,v])], each np array has dim: (4 * (num_other_agents+1),)
        self.episode_record_ = []

        # agents_in_range_record_
        # format: [[list of ids in the range of parametric state vector]]
        self.agents_in_range_record_ = []

    def set_env_config(self, env_config) -> None:
        self.env_config_ = env_config
        if 'num_history_states' not in self.env_config_.keys():
            self.env_config_['num_history_states'] = 0 # default save only current
        if 'ttc_break_tie' not in self.env_config_.keys():
            self.env_config_['ttc_break_tie'] = None # default none
        if 'ego_ttc_break_tie' not in self.env_config_.keys():
            self.env_config_['ego_ttc_break_tie'] = 'random' # default none
        if 'max_num_other_agents_in_range' not in self.env_config_.keys():
            self.env_config_['max_num_other_agents_in_range'] = 25 # default 25

    # only need to call this if using rl agents
    def set_train_config_and_device(self, train_config, device) -> None:
        self.train_config_ = train_config
        self.device_ = device

    def reset(self, use_prev_episode : bool = False, use_saved_agents : Dict[str, Agent] = None, interacting_agents: List[int] = None) -> np.ndarray:
        self.ttc_dps = [] # debug
        # clear episode records!
        del self.episode_record_
        del self.agents_in_range_record_
        self.episode_record_ = []
        self.agents_in_range_record_ = []

        # big matrix to save the parametric states of everyone (in order)
        # over the past 5 timestamps
        # note that this only saves [x,y,cos(th),sin(th),vx,vy]
        N = self.env_config_['max_num_other_agents']+1
        N_0 = self.env_config_['max_num_other_agents_in_range']+1
        self.num_states_to_save_ = (self.env_config_['num_history_states'] + self.env_config_['num_future_states']) * \
                                   self.env_config_['time_gap'] + 1
        self.history_parametric_states_ = np.zeros((N, self.num_states_to_save_, N, self.env_config_['agent_state_dim'])) # dim_0 is when ego_id equals different agent
        self.history_output_parametric_states_ = np.zeros((self.env_config_['stalemate_horizon'], N_0, self.env_config_['agent_state_dim'])) # only support ego_id = 0

        # Note: only one of use_prev_episode and use_saved_agents can be used!
        if (use_prev_episode) and (use_saved_agents is not None):
            raise Exception('only one of use_prev_episode and use_saved_agents can be used!')
        
        if use_saved_agents is not None:
            if len(use_saved_agents) != (1 + self.env_config_['num_other_agents']):
                total_num_agents = 1 + self.env_config_['num_other_agents']
                conditional_log(self.log_name_, logger, f"len(use_saved_agents)={len(use_saved_agents)}, num_other_agents + 1 = {total_num_agents}")
                raise Exception('use_saved_agents length mismatch with env_config')
            self.agents_.clear()
            for input_agent_id, input_agent in use_saved_agents.items():
                # modify the input_agent
                agent = copy.deepcopy(input_agent)
                agent.log_name_ = self.log_name_
                # for rl_agents, only rl_model_path is saved. So we need to recreate the rl_agents for them
                # for non rl_agents, they don't have the attribute: rl_model_path_
                if hasattr(agent, 'rl_model_path_') and agent.rl_model_path_ is not None:
                    if self.train_config_['model'] == 'DDQN':
                        agent.rl_agent_ = DDQNAgent(self.train_config_, self.device_)
                    elif self.train_config_['model'] == 'TwinDDQN':
                        agent.rl_agent_ = TwinDDQNAgent(self.train_config_, self.device_)
                    agent.rl_agent_.load(agent.rl_model_path_)
                    agent.rl_agent_.value_net.eval()
                # This is an insurance. For saved agents in sets they sometimes have the wrong stop_prob. 
                # This is included here to fix that
                agent.stop_prob_ = - 0.5 * agent.b1_ + 0.5
                
                # save the input_agent
                if input_agent_id == '0':
                    self.agents_[input_agent_id] = NeighborhoodV4EgoAgentWrapper(agent, self.env_config_['max_num_other_agents'])
                    conditional_log(self.log_name_, logger, f'Loaded wrapped ego. ego pseudo_id: ' + self.agents_['0'].pseudo_id_, 'debug')
                else:
                    self.agents_[input_agent_id] = NeighborhoodV4DefaultAgentWrapper(agent)

        elif use_prev_episode:
            self.agents_ = copy.deepcopy(self.prev_agents_)
        else:
            self.agents_.clear()

            # === create ego ===
            if ('c1' not in self.env_config_.keys()) or (self.env_config_['c1'] == 0):
                conditional_log(self.log_name_, logger, f'making c0 env left/right turn', 'debug')
                # select ego_path
                select_turn = np.random.randint(2)
                if select_turn == 0:
                    conditional_log(self.log_name_, logger,'left turn')
                    ego_goal_pos = self.__compute_position_from_lane_waypoint('B0', -1)
                    ego_path = ['B3','B10','B0']
                else:
                    conditional_log(self.log_name_, logger,'right turn')
                    ego_goal_pos = self.__compute_position_from_lane_waypoint('B4', -1)
                    ego_path = ['B3','B9','B4']

                lane_order = 0
                waypt_idx = len(self.lane_segments_[ego_path[lane_order]].centerline) - 250 # default fixed if it's not set
                ego_pos = self.__compute_position_from_lane_waypoint(ego_path[lane_order], waypt_idx)
            elif self.env_config_['c1'] == 1:
                conditional_log(self.log_name_, logger, f'making all possible depth-1 paths at t4', 'debug')
                # choose one of the intersection lanes
                t4_lane_ids = intersection_id_dict['t4']
                rand_lane_id = t4_lane_ids[np.random.randint(len(t4_lane_ids))]
                rand_lane_seg = self.lane_segments_[rand_lane_id]
                # expand
                ego_path = [rand_lane_seg.predecessors[0],rand_lane_id,rand_lane_seg.successors[0]]
                # ego pos
                lane_order = 0
                waypt_idx = len(self.lane_segments_[ego_path[lane_order]].centerline) - 250 # default fixed if it's not set
                ego_pos = self.__compute_position_from_lane_waypoint(ego_path[lane_order], waypt_idx)
                # ego goal pos
                ego_goal_pos = self.__compute_position_from_lane_waypoint(ego_path[-1], -1)
            else:
                conditional_log(self.log_name_, logger, f'making all possible paths at random env', 'debug')
                # = c1 =
                ego_pos, ego_goal_pos, ego_path, lane_order, waypt_idx = self.__initialize_ego(self.env_config_['ego_num_intersections_in_path'])
            ego_b1 = 2 * np.random.random() - 1 # [-1,1)
            w, b = self.env_config_['ego_velocity_coeff']
            ego_vel = w * ego_b1 + b
            new_ego_agent = NeighborhoodV4EgoAgent(id='0', observable_state=ObservableState(ego_pos, 0.0, 0.0),
                                                        goal=ObservableState(ego_goal_pos, 0.0, 0.0), 
                                                        curr_lane_id_order=lane_order, # Note: setting this to an earlier lane will let the agent keep going without stopping along lane
                                                        curr_waypoint_idx=waypt_idx,
                                                        path = ego_path,
                                                        default_ttc = self.default_ttc_,
                                                        lane_segments = self.lane_segments_,
                                                        log_name = self.log_name_,
                                                        v_desired = ego_vel,
                                                        b1 = ego_b1)
            self.agents_['0'] = NeighborhoodV4EgoAgentWrapper(new_ego_agent, self.env_config_['max_num_other_agents'])
            conditional_log(self.log_name_, logger, f'ego_path: {str(ego_path)}, lane_order: {lane_order}, waypt_idx: {waypt_idx}, ego pseudo_id: ' + self.agents_['0'].pseudo_id_, 'debug')
            # === create agents ===
            temp_agents = self.generate_other_agents(ego_path, self.env_config_['num_other_agents'])
            if temp_agents is None: # if generating other agents failed
                return None, self.agents_['0'].b1_
            self.agents_.update(temp_agents) # merge the two dictionaries
            # # = end of real stuff =

            # save this set of agent initialization
            self.prev_agents_ = copy.deepcopy(self.agents_)
        
        # TIME
        if self.profiling_:
            curr = time.time()
            print(f'agents init: {curr - self.time_start_}s')

        self.__update_ttc_dp()

        # TIME
        if self.profiling_:
            prev = curr
            curr = time.time()
            print(f'update ttc dp: {curr - prev}s')

        self.curr_ts_ = 0.0
        self.steps_ = 0
        self.ego_vel_sum_ = 0.0

        # if no_ego is True, throw ego to the middle of nowhere and make v_desired_ and velocity = 0
        if ('no_ego' in self.env_config_.keys()) and (self.env_config_['no_ego'] == True):
            ego_pos = Position(-5,-5,0)
            ego_goal_pos = Position(-5,-5,0)
            new_ego_agent = NeighborhoodV4EgoAgent(id='0', observable_state=ObservableState(ego_pos, 0.0, 0.0),
                                                        goal=ObservableState(ego_goal_pos, 0.0, 0.0), 
                                                        curr_lane_id_order=0, # Note: setting this to an earlier lane will let the agent keep going without stopping along lane
                                                        curr_waypoint_idx=0,
                                                        path = [],
                                                        default_ttc = self.default_ttc_,
                                                        lane_segments = self.lane_segments_,
                                                        log_name = self.log_name_,
                                                        v_desired = 5.6,
                                                        b1 = -1.0)
            self.agents_['0'] = NeighborhoodV4EgoAgentWrapper(new_ego_agent, self.env_config_['max_num_other_agents'])
            self.agents_['0'].saved_agent_.velocity_ = 0
            self.agents_['0'].observable_state_.velocity_ = 0

        # NOTE: not used
        self.interacting_agents_ = interacting_agents

        # log
        for agent_id, agent in self.agents_.items():
            if agent_id == '0':
                conditional_log(self.log_name_, logger, f"Ego b1: {agent.b1_}, agg_level: {agent.agg_level_}, v_desired_: {agent.v_desired_}",'debug')
            elif agent.rl_agent_ is not None:
                conditional_log(self.log_name_, logger,
                        f'agent {agent_id} is RL agent with policy {agent.rl_model_path_}', 'debug')
            else:
                conditional_log(self.log_name_, logger, f"agent {agent_id} is default agent with b1: {agent.b1_}, agg_level: {agent.agg_level_}, v_desired_: {agent.v_desired_}",'debug')

        state_vector, ids_in_range = self.__generate_parametric_state()
        self.done_ = False
        self.last_reward_ = None
        self.collision_ = False
        # update self.episode_record_
        self.episode_record_.append((self.__generate_current_episode_record(), 0))
        self.agents_in_range_record_.append(ids_in_range)

        return state_vector, self.agents_['0'].b1_
    
    def generate_other_agents(self, ego_path, num_agents):
        """
        generate num_agents agents, in addition to the agents that are already in self.agents_
        """
        # shuffle agent ids
        curr_num_agents = len(self.agents_)
        ids_order = np.arange(curr_num_agents,curr_num_agents + num_agents)
        min_id = curr_num_agents
        if self.env_config_['agent_shuffle_ids'] == True:
            np.random.shuffle(ids_order)
        conditional_log(self.log_name_, logger, f'ids_order: {str(ids_order)}', 'debug')
        # rl agent id assignment
        agent_model_assignment = self.__assign_agent_models(min_id, num_agents)

        expanded_lane_set = self.__expand_lane_set(ego_path, self.env_config_['expanded_lane_set_depth'])
        conditional_log(self.log_name_, logger, 'expanded_lane_set_depth=' + str(self.env_config_['expanded_lane_set_depth']))
        temp_agent = None
        temp_agents = {}
        generate_agents_success = False
        generate_agents_retry = 0
        prev_depth = self.env_config_['expanded_lane_set_depth']

        w, b = self.env_config_['agent_velocity_coeff']

        while not generate_agents_success:
            generate_agents_retry += 1
            for i in range(num_agents):
                id = str(ids_order[i])
                generated = False
                num_retry = 0
                while not generated and (num_retry < self.max_retry_):
                    num_retry += 1
                    # randomly select a lane and waypoint in expanded_lane_set. if collides with someone else, keep doing until you don't collide
                    random_idx = np.random.randint(len(expanded_lane_set))
                    lane_id = list(expanded_lane_set)[random_idx]
                    waypoint_idx = np.random.randint(len(self.lane_segments_[lane_id].centerline))
                    agent_pos = self.__compute_position_from_lane_waypoint(lane_id, waypoint_idx)
                    
                    # plan a path from this lane within expanded_lane_set
                    # set path and goal
                    agent_path = self.__generate_path_in_set(lane_id, expanded_lane_set)
                    # we let the agent drive around the whole map and return to current location after it finishes the current path
                    agent_extending_path = self.generate_agent_extending_path(agent_path[-1])
                    agent_path = agent_path + agent_extending_path[1:] # we want to get rid of the repeating lane_id at start of agent_extending_path
                    agent_goal_pos = self.__compute_position_from_lane_waypoint(agent_path[-1], -1)
                    
                    # velocity
                    agent_b1 = 2 * np.random.random() - 1 # [-1,1)
                    agent_vel = w * agent_b1 + b

                    if agent_model_assignment[i] == -1:
                        # create default agent
                        temp_agent = NeighborhoodV4DefaultAgent(id=id, observable_state=ObservableState(agent_pos, 0.0, 0.0),
                                                        goal=ObservableState(agent_goal_pos, 0.0, 0.0), 
                                                        curr_lane_id_order=0, # Note: setting this to an earlier lane will let the agent keep going without stopping along lane
                                                        curr_waypoint_idx=waypoint_idx,
                                                        path = agent_path,
                                                        default_ttc = self.default_ttc_,
                                                        lane_segments = self.lane_segments_,
                                                        stochastic_stop = self.env_config_['agent_stochastic_stop'],
                                                        log_name = self.log_name_,
                                                        v_desired = agent_vel,
                                                        b1 = agent_b1)
                    else:
                        # create RL agent
                        temp_agent = NeighborhoodV4DefaultAgent(id=id, observable_state=ObservableState(agent_pos, 0.0, 0.0),
                                                        goal=ObservableState(agent_goal_pos, 0.0, 0.0), 
                                                        curr_lane_id_order=0, # Note: setting this to an earlier lane will let the agent keep going without stopping along lane
                                                        curr_waypoint_idx=waypoint_idx,
                                                        path = agent_path,
                                                        default_ttc = self.default_ttc_,
                                                        lane_segments = self.lane_segments_,
                                                        stochastic_stop = self.env_config_['agent_stochastic_stop'],
                                                        log_name = self.log_name_,
                                                        rl_model_path = self.env_config_['rl_agent_configs'][agent_model_assignment[i]][0], 
                                                        train_configs = self.train_config_,
                                                        device = self.device_,
                                                        v_desired = agent_vel,
                                                        b1 = agent_b1)
                    
                    # temp_agents have every other agent that has been initialized
                    if self.__check_any_collision(temp_agent, self.agents_) or self.__check_any_collision(temp_agent, temp_agents):
                        continue

                    if self.__meets_spacing_requirement(temp_agent, self.agents_) and self.__meets_spacing_requirement(temp_agent, temp_agents):
                        temp_agents[id] = NeighborhoodV4DefaultAgentWrapper(temp_agent)
                        generated = True

                    # conditional_log(self.log_name_, logger, f'agent {id} path: {str(agent_path)}, goal: {self.lane_segments_[agent_path[-1]].centerline[-1,:]}')``
                
                # if this agent is not successfully initialized, break out of for loop
                if not generated:
                    conditional_log(self.log_name_, logger, f'agent{id} generation unsuccessful. Restart agent initialization...')
                    break

            if len(list(temp_agents.keys())) == num_agents:
                conditional_log(self.log_name_, logger, 'Successfully generated all agents')
                generate_agents_success = True
            else:
                temp_agents = {}
                if generate_agents_retry >= 3:
                    prev_depth += 1
                    if prev_depth >= 9:
                        conditional_log(self.log_name_, logger, f'lane set depth = {prev_depth}, exceeding upper limit. Reset fails.')
                        return None
                    conditional_log(self.log_name_, logger, f'generate agents 3 failures. expanding lane set depth to {prev_depth}')
                    expanded_lane_set = self.__expand_lane_set(ego_path, prev_depth)
                    if (len(expanded_lane_set) == len(self.lane_segments_)): # if expanded_lane_set has covered all lanes
                        conditional_log(self.log_name_, logger, f'lane set depth = {prev_depth}, exceeding upper limit. Reset fails.')
                        return None
                    generate_agents_retry = 0
        
        return temp_agents

    def convert_dict_to_agent(self, agent_init_dict: dict, is_ego: bool, agent_id: str) -> Agent:
        agent_lane_id = agent_init_dict['path'][agent_init_dict['curr_lane_id_order']]
        # process negative indexing
        if agent_init_dict['curr_waypoint_idx'] < 0:
            agent_init_dict['curr_waypoint_idx'] = len(self.lane_segments_[agent_lane_id].centerline) + agent_init_dict['curr_waypoint_idx']
        agent_pos = self.__compute_position_from_lane_waypoint(agent_lane_id, agent_init_dict['curr_waypoint_idx'])
        agent_goal_pos = self.__compute_position_from_lane_waypoint(agent_init_dict['path'][-1], -1)

        if is_ego:
            # specify b1 and v_desired. Have velocity_coeff fixed! Double check with velocity_coeff
            # if you change ego_velocity_coeff, you basically have to change the whole dataset (interaction set, collision set, etc)
            w, b = self.env_config_['ego_velocity_coeff']
            assert agent_init_dict['v_desired'] == w * agent_init_dict['b1'] + b
            temp_agent = NeighborhoodV4EgoAgent(id='0', observable_state=ObservableState(agent_pos, 0.0, 0.0),
                                                            goal=ObservableState(agent_goal_pos, 0.0, 0.0), 
                                                            curr_lane_id_order=agent_init_dict['curr_lane_id_order'], # Note: setting this to an earlier lane will let the agent keep going without stopping along lane
                                                            curr_waypoint_idx=agent_init_dict['curr_waypoint_idx'],
                                                            path = agent_init_dict['path'],
                                                            default_ttc = self.default_ttc_,
                                                            lane_segments = self.lane_segments_,
                                                            log_name = self.log_name_,
                                                            v_desired = agent_init_dict['v_desired'],
                                                            b1 = agent_init_dict['b1'])
            temp_agent_wrapped = NeighborhoodV4EgoAgentWrapper(temp_agent, self.env_config_['max_num_other_agents'])
        else:
            # specify b1 and v_desired. Have velocity_coeff fixed! Double check with velocity_coeff
            # if you change ego_velocity_coeff, you basically have to change the whole dataset (interaction set, collision set, etc)
            w, b = self.env_config_['agent_velocity_coeff']
            assert agent_init_dict['v_desired'] == w * agent_init_dict['b1'] + b
            temp_agent = NeighborhoodV4DefaultAgent(id=agent_id, 
                                                observable_state=ObservableState(agent_pos, 0.0, 0.0),
                                                goal=ObservableState(agent_goal_pos, 0.0, 0.0), 
                                                curr_lane_id_order=agent_init_dict['curr_lane_id_order'], # Note: setting this to an earlier lane will let the agent keep going without stopping along lane
                                                curr_waypoint_idx=agent_init_dict['curr_waypoint_idx'],
                                                path = agent_init_dict['path'],
                                                default_ttc = self.default_ttc_,
                                                lane_segments = self.lane_segments_,
                                                stochastic_stop = self.env_config_['agent_stochastic_stop'],
                                                log_name = self.log_name_,
                                                v_desired = agent_init_dict['v_desired'],
                                                b1 = agent_init_dict['b1'])
            temp_agent_wrapped = NeighborhoodV4DefaultAgentWrapper(temp_agent)
        return temp_agent_wrapped

    def convert_dict_to_agents(self, ego_init_dict: dict, agent_init_dicts: List[dict]) -> Dict[str, Agent]:
        """
        Given an init_dict for ego and a list of init_dict for agents, output a dictionary of agents
        which could be plugged in for env reset()
        ego_init_dict example: {'path': ['B3', 'B10', 'B0'], 'curr_lane_id_order': 0, 'curr_waypoint_idx': -250}
        agent_init_dict example: {'path': ['B3', 'B10', 'B0'], 'curr_lane_id_order': 0, 'curr_waypoint_idx': -250, 'agg_level': 0, (Optional: 'v_desired': 5.6)}
        - negative waypoint indexing is supported
        """
        agents_dict = {}
        # == ego ==
        agents_dict['0'] = self.convert_dict_to_agent(ego_init_dict, True, '0')
        # == agents ==
        for i in range(len(agent_init_dicts)): # use index instead of for-each loop to keep ordering
            agent_id = str(i+1)
            agents_dict[agent_id] = self.convert_dict_to_agent(agent_init_dicts[i], False, agent_id)

        return agents_dict
    
    def step(self, action) -> (np.ndarray, float, bool, []):
        """
        action: discrete. 0 for stop, 1 for go
        """
        self.curr_ts_ += self.dt_
        self.steps_ += 1

        # TIME
        if self.profiling_:
            curr = time.time()
        
        ttc_trackers = None
        
        # only step forward if the system is not done yet
        if not self.done_:
            # update ttc for everyone
            self.__update_ttc_dp() # O(2n^2)

            # TIME
            if self.profiling_:
                prev = curr
                curr = time.time()
                print(f'update ttc dp: {curr - prev}s')

            # so when we call this function, we just need to take values
            for i in range(1 + self.env_config_['num_other_agents']):
                agent_id = str(i)
                self.agents_[agent_id].calculate_ttc(self.agents_, self.dt_, self.ttc_dp_)

            # TIME
            if self.profiling_:
                prev = curr
                curr = time.time()
                print(f'update ttc: {curr - prev}s')

            # stop_for_graph is a directed graph: {vertex: [(neighbor_vertex, weight)]}
            stop_for_graph = {}

            # DEBUG
            ego_should_stop = None
            if ('no_ego' in self.env_config_.keys()) and (self.env_config_['no_ego'] == True):
                action = 0
                stop_for_graph[0] = []
                conditional_log(self.log_name_, logger, "No-ego model. Ego action = 0")
            else:
                if ('use_default_ego' in self.env_config_.keys()) and self.env_config_['use_default_ego']:
                    if ('ego_baseline' in self.env_config_.keys()) and (self.env_config_['ego_baseline'] is not None):
                        ego_should_stop, ids_should_stop_for, ttc_trackers = self.agents_['0'].should_stop_wrapped(self.dt_, 
                            self.agents_, self.ttc_dp_, self.env_config_['ego_baseline'], 
                            self.env_config_['include_agents_within_range'], ttc_break_tie=self.env_config_['ego_ttc_break_tie'], 
                            return_ttc_tracker=True)
                    else:
                        ego_should_stop, ids_should_stop_for, ttc_trackers = self.agents_['0'].should_stop_wrapped(self.dt_, 
                            self.agents_, self.ttc_dp_, ttc_break_tie=self.env_config_['ego_ttc_break_tie'], return_ttc_tracker=True)
                    stop_for_graph[0] = ids_should_stop_for
                    if ego_should_stop:
                        action = 0
                        conditional_log(self.log_name_, logger, "Default ego action = 0")
                    else:
                        action = 1
                        conditional_log(self.log_name_, logger, "Default ego action = 1")
                else:
                    _, _, ttc_trackers = self.agents_['0'].should_stop_wrapped(self.dt_, 
                            self.agents_, self.ttc_dp_, self.env_config_['ego_baseline'], 
                            self.env_config_['include_agents_within_range'], ttc_break_tie=self.env_config_['ego_ttc_break_tie'], 
                            return_ttc_tracker=True)
                    stop_for_graph[0] = []

            
            # get a snapshot of the world w.r.t. every agent
            agent_states, agent_actions,_ = self.generate_agent_states(reward_and_done=False)
            # step agents based on action
            conditional_log(self.log_name_, logger,
                            'agent_baseline:' + str(self.env_config_['agent_baseline']) + \
                            ' ttc_break_tie:'  + self.env_config_['ttc_break_tie'] + 'ego_ttc_break_tie: ' + self.env_config_['ego_ttc_break_tie'], 'debug') # DEBUG
            for id in range(1 + self.env_config_['num_other_agents']):
                agent_id = str(id)
                if agent_id == '0':
                    # ego_action_noise
                    if 'ego_action_noise' in self.env_config_.keys() and self.env_config_['ego_action_noise'] > 0:
                        ego_action_noise_sample = np.random.random()
                        if (ego_action_noise_sample <= self.env_config_['ego_action_noise']):
                            ego_action_noise = self.env_config_['ego_action_noise']
                            action = int(1 - action)
                            conditional_log(self.log_name_, logger,
                                f'ego switch action to {action} given ego_action_noise={ego_action_noise}', 'debug') # DEBUG
                            
                    # step ego
                    self.curr_action_ = action
                    if action == 1:
                        self.ego_vel_sum_ += self.agents_['0'].v_desired_
                        self.agents_['0'].step(1, self.dt_)
                    else:
                        self.agents_['0'].step(0, self.dt_)
                else:
                    ids_should_stop_for = self.agents_[agent_id].step(self.dt_, self.agents_, self.ttc_dp_, agent_states,\
                        agent_action_noise = self.env_config_['agent_action_noise'],\
                        ttc_break_tie=self.env_config_['ttc_break_tie'],\
                        agent_baseline=self.env_config_['agent_baseline'])
                    stop_for_graph[int(agent_id)] = ids_should_stop_for

            # detect cycle in stop_for_graph
            if ('stalemate_breaker' in self.env_config_.keys()) and (self.env_config_['stalemate_breaker']):
                is_cycle, cycle = self.__find_cycle(stop_for_graph)
                if is_cycle:
                    cycle.reverse()
                    cycle.append(cycle[0])
                    max_edge_weight = -float('Inf')
                    max_vertex = -1
                    for i in range(len(cycle) - 1):
                        for neighbor, weight in stop_for_graph[cycle[i]]:
                            if neighbor == cycle[i+1] and weight > max_edge_weight:
                                max_edge_weight = weight
                                # avoid forcing ego forward
                                if cycle[i] == 0:
                                    max_vertex = cycle[i+1]
                                else:
                                    max_vertex = cycle[i]
                    conditional_log(self.log_name_, logger, f"max_edge_weight={max_edge_weight}, max_vertex={max_vertex}. Forcing {max_vertex} to step forward...")
                    # force one of the agents with the largest edge to step forward
                    _ = self.agents_[str(max_vertex)].force_action(1, self.dt_)

            # TIME
            if self.profiling_:
                prev = curr
                curr = time.time()
                print(f'step agents: {curr - prev}s')

        # calculate single step reward and done
        parametric_state, ids_in_range = self.__generate_parametric_state() # O(n)
        conditional_log(self.log_name_, logger, f'ids_in_range: {ids_in_range}', 'debug')
        if not self.done_: # only update self.done_ if the episode is not done yet
            reward, done, collision = self.__calculate_reward_and_done() # O(n)
            self.done_ = done
            self.last_reward_ = reward
            self.collision_ = collision
        else:
            reward = 0

        # TIME
        if self.profiling_:
            prev = curr
            curr = time.time()
            print(f'calculate reward and done and state: {curr - prev}s')
        
        # update self.episode_record_
        self.episode_record_.append((self.__generate_current_episode_record(), reward))
        self.agents_in_range_record_.append(ids_in_range)

        # return average velocity, collision flag, distance driven, ep_record, ego b1
        info = [self.ego_vel_sum_ / self.steps_, self.collision_, \
                self.agents_['0'].dist_driven_, self.episode_record_, \
                self.agents_['0'].b1_, self.agents_in_range_record_, ttc_trackers, action]

        return parametric_state, self.last_reward_, self.done_, info 

    def generate_agent_states(self, reward_and_done=True) -> (np.ndarray, np.ndarray, np.ndarray):
        # save a deepcopy snapshot of current agent states
        curr_agents = copy.deepcopy(self.agents_)
        
        agent_states = []
        agent_rewards = []
        agent_actions = []
        agent_dones = []
        agent_b1s = []

        for i in range(1, self.env_config_['num_other_agents']+1):
            id = str(i)
            # b1s
            agent_b1s.append(self.agents_[id].b1_)
            # swap agent and ego in self.agents_
            self.agents_[id] = curr_agents['0']
            self.agents_['0'] = curr_agents[id]
            # state
            parametric_state, _ = np.array(self.__generate_parametric_state(ego_id = i))
            agent_states.append(parametric_state)
            # reward and done
            if reward_and_done:
                reward, done, collision = self.__calculate_reward_and_done(ego_id = i)
                agent_rewards.append(reward)
                agent_dones.append(done)
            # actions
            agent_actions.append(int(curr_agents[id].observable_state_.velocity_ > 0))
            # swap back
            self.agents_[id] = curr_agents[id]
            self.agents_['0'] = curr_agents['0']
        
        if reward_and_done:
            return agent_states, agent_rewards, agent_actions, agent_dones, agent_b1s
        else:
            return agent_states, agent_actions, agent_b1s

    def render(self, save_fig_dir = '', mode='human', close=False):
        # 1. draw all lane centers and drivable zones
        logger.debug('Rendering map... ')
        fig = plt.figure(figsize=(7,8))
        ax = fig.add_subplot(8,1,(1,7))
        ax.set_xlim([-2, self.max_xy_value_+2])
        ax.set_ylim([-2, self.max_xy_value_+2])

        for lane_segment in list(lane_segments.values()):
            lane_cl = lane_segment.centerline
            ax.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="grey", alpha=1, linewidth=1, zorder=0)

        # 2. for i = 0 to n: draw agents
        logger.debug('Rendering all agents on map... ')
        for agent in self.agents_.values():
            visualize_agent(ax, agent)
        
        # 3. draw future horizon waypoints
        # for agent in self.agents_.values():
        #     ax.plot(agent.future_horizon_positions_[0,:,0], agent.future_horizon_positions_[0,:,1], color="k", alpha=1, linewidth=2, zorder=0)
        #     ax.plot(agent.future_horizon_positions_[1,:,0], agent.future_horizon_positions_[1,:,1], color="m", alpha=1, linewidth=2, zorder=0)

        # 4. draw agent actions
        ax = fig.add_subplot(8,1,8)
        agent_actions = []
        for i in range(self.env_config_['num_other_agents']+1):
            id = str(i)
            vel = self.agents_[id].observable_state_.velocity_
            agent_actions.append(1 if vel > 0 else 0)
        ax.bar(np.arange(self.env_config_['num_other_agents']+1), agent_actions)
        ax.set_xticks(np.arange(self.env_config_['num_other_agents']+1))
        ax.set_xlabel('agent_id')
        ax.set_ylabel('action')

        plt.tight_layout()

        # save fig
        if save_fig_dir != '':
            if not Path(f"{save_fig_dir}").exists():
                os.makedirs(f"{save_fig_dir}")
            plt.savefig(
                f"{save_fig_dir}/{int(self.curr_ts_*10):04d}.png",
                dpi=400,
            )
            plt.close()
        else:
            # self.fig_ = fig
            plt.draw()
            plt.pause(0.1)
            plt.close()

        return None

    def __generate_parametric_state(self, ego_id=0) -> np.ndarray:
        ids_to_include = []
        # save the current states of all agents
        curr_all_agents_states = np.zeros((1,self.env_config_['max_num_other_agents']+1,self.env_config_['agent_state_dim']))
        # note that this only saves [x,y,cos(th),sin(th),vx,vy]
        ego_position = self.agents_['0'].observable_state_.position_
        ego_xy = np.array([ego_position.x_, ego_position.y_])
        ego_velocity_vector_city_frame = \
                    np.array([math.cos(ego_position.heading_), math.sin(ego_position.heading_)]) \
                    * self.agents_['0'].observable_state_.velocity_
        if not self.env_config_['use_global_frame']:
            # use ego-centric frame. (ego state is also included)
            city_frame_to_ego_frame_se3 = self.__calculate_city_frame_to_ego_frame_se3()
            for i in range(len(self.agents_.keys())):
                id = str(i)
                # calculate poly_dist first and decide whether to include this agent based on poly_dist
                poly_dist = calculate_poly_distance(self.agents_['0'], self.agents_[id])
                include_agent = True # default include everyone
                if ('include_agents_within_range' in self.env_config_.keys()) and (self.env_config_['include_agents_within_range'] > 0):
                    include_agent = (poly_dist <= self.env_config_['include_agents_within_range'])
                # now actually calculate the state
                if include_agent:
                    ids_to_include.append(i)
                    curr_agent_state = []

                    agent_position = self.agents_[id].observable_state_.position_
                    # convert position to ego frame
                    agent_xy = np.array([[agent_position.x_, agent_position.y_, 0]])
                    agent_xy_ego_frame = city_frame_to_ego_frame_se3.transform_point_cloud(agent_xy)[0,:2]
                    # convert speed to ego frame
                    agent_velocity_vector_city_frame = \
                        np.array([math.cos(agent_position.heading_), math.sin(agent_position.heading_)]) \
                        * self.agents_[id].observable_state_.velocity_
                    relative_velocity_vector = agent_velocity_vector_city_frame - ego_velocity_vector_city_frame
                    relative_velocity_vector = np.hstack((relative_velocity_vector + ego_xy, 0)).reshape((1,-1)) # put it at the origin of ego so the transformation will be correct
                    relative_velocity_vector_ego_frame = city_frame_to_ego_frame_se3.transform_point_cloud(relative_velocity_vector)[0,:2] 
                    # convert heading to ego frame
                    heading_ego_frame = wrap_to_pi(agent_position.heading_ - ego_position.heading_)
                    # global frame range definition for normalization
                    xy_range = [-self.max_xy_value_, self.max_xy_value_]
                    vxvy_range = [-2*self.max_rel_v_value_, 2*self.max_rel_v_value_] # 2 * max_relative_vel
                    target_range = [-1,1]
                
                    if self.env_config_['normalize']:
                        norm_xy = remap(agent_xy_ego_frame, xy_range, target_range)
                        norm_vxvy = remap(relative_velocity_vector_ego_frame, xy_range, target_range)
                        curr_agent_state += [norm_xy[0], norm_xy[1], # x,y
                                            norm_vxvy[0], norm_vxvy[1], # vx, vy
                                            math.cos(heading_ego_frame), math.sin(heading_ego_frame)] # cos(th), sin(th)
                    else:
                        curr_agent_state += [agent_xy_ego_frame[0], agent_xy_ego_frame[1], # x,y
                                            relative_velocity_vector_ego_frame[0], relative_velocity_vector_ego_frame[1], # vx, vy
                                            math.cos(heading_ego_frame), math.sin(heading_ego_frame)] # cos(th), sin(th)
                    # 10 future waypoints
                    if 'include_future_waypoints' in self.env_config_.keys() and self.env_config_['include_future_waypoints']:
                        future_reference_route = self.__find_future_reference_route(id, self.env_config_['include_future_waypoints'])
                        if self.env_config_['normalize']:
                            norm_future_reference_route = remap(future_reference_route, xy_range, target_range)
                            curr_agent_state += list(norm_future_reference_route.flatten())
                        else:
                            curr_agent_state += list(future_reference_route.flatten())
                    # append ttc
                    if 'include_ttc' in list(self.env_config_.keys()) and self.env_config_['include_ttc']:
                        curr_agent_state += self.__generate_ttc_values(i, ego_id=ego_id).tolist()
                    # ego-agent polygon distance
                    if 'include_polygon_dist' in self.env_config_.keys() and self.env_config_['include_polygon_dist']:
                        poly_dist = 2. / (1. + math.exp(-poly_dist)) - 1. # rescale to [0,1) cuz poly_dist >= 0
                        curr_agent_state.append(poly_dist) # polygon_dist
                    # save curr_agent_state to curr_all_agents_states
                    curr_all_agents_states[0,i,:] = np.array(curr_agent_state)
        else:
            # use global frame. [ego state, agent state]
            if self.env_config_['normalize']:
                xy_range = [0,self.max_xy_value_]
                vxvy_range = [-2*self.max_v_value_,2*self.max_v_value_] # 2 * max_vel
                target_range = [-1,1]
                for i in range(1 + self.env_config_['num_other_agents']):
                    id = str(i)
                    # calculate poly_dist first and decide whether to include this agent based on poly_dist
                    poly_dist = calculate_poly_distance(self.agents_['0'], self.agents_[id])
                    include_agent = True
                    if ('include_agents_within_range' in self.env_config_.keys()) and (self.env_config_['include_agents_within_range'] > 0):
                        include_agent = (poly_dist <= self.env_config_['include_agents_within_range'])
                    # now actually calculate the state
                    if include_agent:
                        ids_to_include.append(i)
                        curr_agent_state = []

                        norm_x = remap(self.agents_[id].observable_state_.position_.x_, xy_range, target_range)
                        norm_y = remap(self.agents_[id].observable_state_.position_.y_, xy_range, target_range)
                        theta = self.agents_[id].observable_state_.position_.heading_
                        vx = self.agents_[id].observable_state_.velocity_ * math.cos(theta)
                        vy = self.agents_[id].observable_state_.velocity_ * math.sin(theta)
                        norm_vx = remap(vx, vxvy_range, target_range)
                        norm_vy = remap(vy, vxvy_range, target_range)
                        curr_agent_state += [norm_x, norm_y, norm_vx, norm_vy, math.cos(theta), math.sin(theta)]
                        # 10 future waypoints
                        if 'include_future_waypoints' in self.env_config_.keys() and self.env_config_['include_future_waypoints'] > 0:
                            future_reference_route = self.__find_future_reference_route(id, self.env_config_['include_future_waypoints'])
                            norm_future_reference_route = remap(future_reference_route, xy_range, target_range)
                            curr_agent_state += list(norm_future_reference_route.flatten())
                        # append ttc
                        if 'include_ttc' in list(self.env_config_.keys()) and self.env_config_['include_ttc']:
                            curr_agent_state += self.__generate_ttc_values(i, ego_id=ego_id).tolist()
                        # ego-agent polygon distance
                        if 'include_polygon_dist' in self.env_config_.keys() and self.env_config_['include_polygon_dist']:
                            poly_dist = 2. / (1. + math.exp(-poly_dist)) - 1. # rescale to [0,1) cuz poly_dist >= 0
                            curr_agent_state.append(poly_dist) # polygon_dist
                        # save curr_agent_state to curr_all_agents_states
                        curr_all_agents_states[0,i,:] = np.array(curr_agent_state)
            else:
                for i in range(1 + self.env_config_['num_other_agents']):
                    id = str(i)
                    # calculate poly_dist first and decide whether to include this agent based on poly_dist
                    poly_dist = calculate_poly_distance(self.agents_['0'], self.agents_[id])
                    include_agent = True
                    if ('include_agents_within_range' in self.env_config_.keys()) and (self.env_config_['include_agents_within_range'] > 0):
                        include_agent = (poly_dist <= self.env_config_['include_agents_within_range'])
                    # now actually calculate the state
                    if include_agent:
                        ids_to_include.append(i)
                        curr_agent_state = []

                        theta = self.agents_[id].observable_state_.position_.heading_
                        vx = self.agents_[id].observable_state_.velocity_ * math.cos(theta)
                        vy = self.agents_[id].observable_state_.velocity_ * math.sin(theta)
                        curr_agent_state += [self.agents_[id].observable_state_.position_.x_,
                                            self.agents_[id].observable_state_.position_.y_,
                                            vx, vy, math.cos(theta), math.sin(theta)]
                        # 10 future waypoints
                        if 'include_future_waypoints' in self.env_config_.keys() and self.env_config_['include_future_waypoints'] > 0:
                            future_reference_route = self.__find_future_reference_route(id, self.env_config_['include_future_waypoints'])
                            curr_agent_state += list(future_reference_route.flatten())
                        # append ttc
                        if 'include_ttc' in list(self.env_config_.keys()) and self.env_config_['include_ttc']:
                            curr_agent_state += self.__generate_ttc_values(i, ego_id=ego_id).tolist()
                        # ego-agent polygon distance
                        if 'include_polygon_dist' in self.env_config_.keys() and self.env_config_['include_polygon_dist'] > 0:
                            poly_dist = 2. / (1. + math.exp(-poly_dist)) - 1. # rescale to [0,1) cuz poly_dist >= 0
                            curr_agent_state.append(poly_dist) # polygon_dist
                        # save curr_agent_state to curr_all_agents_states
                        curr_all_agents_states[0,i,:] = np.array(curr_agent_state)

        # DEBUG
        if 'include_agents_within_range' in self.env_config_.keys() and self.env_config_['include_agents_within_range'] > 0:
            # num_agents_in_state = int(len(parametric_state) / self.env_config_['agent_state_dim'])
            num_agents_in_state = len(ids_to_include)
            # conditional_log(self.log_name_, logger, f'{num_agents_in_state} agents in state: {ids_to_include}', 'info')

        # update self.history_parametric_states_
        if self.steps_ == 0:
            self.history_parametric_states_[ego_id,:,:,:] = np.tile(curr_all_agents_states, (self.num_states_to_save_, 1, 1))
        else:
            self.history_parametric_states_[ego_id,:,:,:] = np.concatenate((self.history_parametric_states_[ego_id,1:,:,:], curr_all_agents_states), axis=0)
        # now we construct the parametric_state to output
        parametric_state = np.zeros((1, self.env_config_['max_num_other_agents_in_range'] + 1, self.env_config_['agent_total_state_dim']))
        ts_to_include = np.flip(- np.arange(self.env_config_['num_ts_in_state']) * self.env_config_['time_gap'] - 1)
        for i in range(len(ids_to_include)):
            curr_id = ids_to_include[i]
            parametric_state[0,i,:] = self.history_parametric_states_[ego_id, ts_to_include, curr_id, :].flatten()
        if ego_id == 0:
            self.history_output_parametric_states_ = np.concatenate((self.history_output_parametric_states_[1:,:,:], parametric_state[:,:,-(self.env_config_['agent_state_dim']):]), axis=0)

        return parametric_state.flatten(), ids_to_include

    def __generate_ttc_values(self, agent_id: int, ego_id: int=0) -> np.ndarray:
        # return a list of [ttc0, ttc1, ttc2, ttc3] from ego to this agent i
        ttc_values = np.array([self.ttc_dp_[0,ego_id,agent_id], self.ttc_dp_[1,ego_id,agent_id], self.ttc_dp_[2,ego_id,agent_id], self.ttc_dp_[3,ego_id,agent_id]])
        ttc_values = 2. / (1. + np.exp(-ttc_values)) - 1. # rescale to [0,1) cuz ttc_values >= 0
        return ttc_values

    def __update_ttc_dp(self) -> None:
        # all default ttc values
        N = 1 + self.env_config_['num_other_agents'] # for the sake of readability
        H = self.agents_['0'].ttc_horizon_ # for the sake of readability
        R = self.agents_['0'].radius_ # for the sake of readability
        ttc_dp_raw = np.ones((N, N, 4)) * self.default_ttc_
        # initialize big_temp_matrix
        agents_fhp1 = np.zeros((0, 4, H, 2))
        agents_fhp2 = np.zeros((0, 4, H, 2))
        for i in range(N):
            # construct the 2 future horizon waypoints matrices
            id = str(i)
            # fhp = future_horizon_positions_
            fhp_idx_curr_v = 1 if self.agents_[id].observable_state_.velocity_ > 0 else 0
            fhp_raw = self.agents_[id].future_horizon_positions_
            agent_fhp1 = np.vstack((np.expand_dims(fhp_raw[fhp_idx_curr_v, :, :], axis=0),
                                    np.expand_dims(fhp_raw[0, :, :], axis=0),
                                    np.expand_dims(fhp_raw[1, :, :], axis=0),
                                    np.expand_dims(fhp_raw[1, :, :], axis=0)))
            agent_fhp2 = np.vstack((np.expand_dims(fhp_raw[fhp_idx_curr_v, :, :], axis=0),
                                    np.expand_dims(fhp_raw[1, :, :], axis=0),
                                    np.expand_dims(fhp_raw[0, :, :], axis=0),
                                    np.expand_dims(fhp_raw[1, :, :], axis=0)))           
            # fill into big_temp_matrix
            agents_fhp1 = np.concatenate((agents_fhp1, np.expand_dims(agent_fhp1,0)), axis=0)
            agents_fhp2 = np.concatenate((agents_fhp2, np.expand_dims(agent_fhp2,0)), axis=0)
        
        agents_fhp1 = np.repeat(agents_fhp1, N, axis=0) # (N*N)*4*ttc_horizon*2
        agents_fhp2 = np.tile(agents_fhp2, (N,1,1,1)) # (N*N)*4*ttc_horizon*2
        # do calculations with big_temp_matrix
        dxdy = (agents_fhp1 - agents_fhp2).reshape((N,N,4,H,2)) # containing dx, dy
        norms = la.norm(dxdy, axis=4) # N*N*4*H
        for i in range(N):
            norms[i,i,:,:] = 10.0 # prevent same-agent pair from being identified as collision
        # print("big_temp_matrix norms: ", big_temp_matrix)
        is_collision = (norms < 2 * R + 1.0) # 3 = 2r + 1 # N*N*4*H [0,0,0,1,1...]
        # print("big_temp_matrix is_collision: ", big_temp_matrix)
        ttc_raw = np.argmax(is_collision, axis=3) # N*N*4
        no_collision_mask = np.all(is_collision == False, axis=3) # N*N*4
        ttc_dp_new = np.where(no_collision_mask, ttc_dp_raw, (ttc_raw+1)/10.0)
        ttc_dp_new = np.transpose(ttc_dp_new, (2,0,1)) # 4*N*N

        # print("ttc_dp old: ", ttc_dp)
        # print("ttc_dp new: ", ttc_dp_new)
        # if not np.all(ttc_dp - ttc_dp_new < 1e-5):
        #     pdb.set_trace()
        # assert np.all(ttc_dp - ttc_dp_new < 1e-5)
        
        self.ttc_dp_ = ttc_dp_new
        self.ttc_dps.append(ttc_dp_new)
    
    def __find_future_reference_route(self, agent_id: str, num_waypoints: int) -> List[float]:
        """
        find num_waypoints points along the planned route into the future
        """
        # find the num_waypoints subsequent waypoints (every 10 waypoints apart)
        total_remain = num_waypoints
        future_reference_route = []

        # special case: if len(path) == 0
        if len(self.agents_[agent_id].path_) == 0:
            agent_x = self.agents_[agent_id].observable_state_.position_.x_
            agent_y = self.agents_[agent_id].observable_state_.position_.y_
            for i in range(num_waypoints):
                future_reference_route.append(np.array([agent_x, agent_y]))
            return np.array(future_reference_route)
        
        curr_lane_id_order = self.agents_[agent_id].closest_lane_id_order_
        curr_waypoint_idx = self.agents_[agent_id].closest_waypoint_idx_
        past_last_lane = False

        while total_remain > 0 and (curr_lane_id_order < len(self.agents_[agent_id].path_)):
            remain = 10
            while remain > 0:
                # if closest_lane_id_order >= exceeding path length, there's no future waypoint
                if curr_lane_id_order >= len(self.agents_[agent_id].path_):
                    past_last_lane = True
                    break
                curr_lane_id = self.agents_[agent_id].path_[curr_lane_id_order]
                while remain > 0:
                    curr_waypoint_idx += 1
                    remain -= 1
                    # if we run out of waypoints in the current lane, update lane_id_order and exit inner loop
                    if (curr_waypoint_idx == len(lane_segments[curr_lane_id].centerline)):
                        curr_lane_id_order += 1
                        curr_waypoint_idx = 0
                        break
            if not past_last_lane:
                future_reference_route.append(self.lane_segments_[curr_lane_id].centerline[curr_waypoint_idx,:])
                total_remain -= 1

        # edge case: if there's < 10 waypoints left:
        while total_remain > 0:
            curr_lane_id = self.agents_[agent_id].path_[-1]
            last_waypoint = self.lane_segments_[curr_lane_id].centerline[-1,:]
            future_reference_route.append(last_waypoint)
            total_remain -= 1

        future_reference_route = np.array(future_reference_route)
        # convert to ego frame
        if not self.env_config_['use_global_frame']:
            future_reference_route = np.hstack((future_reference_route, np.zeros((len(future_reference_route),1))))
            city_frame_to_ego_frame_se3 = self.__calculate_city_frame_to_ego_frame_se3()
            future_reference_route = city_frame_to_ego_frame_se3.transform_point_cloud(future_reference_route)[:,:2]
        # [x1,y1; x2,y2; ...; x10,y10]
        return future_reference_route

    def __calculate_reward_and_done(self, ego_id=0) -> Tuple[float, bool, bool]:
        # calculate single step reward
        done = False
        collision = False
        reward = 0.0
        ego_b1 = self.agents_['0'].b1_ # for the sake of readability
        # in no_ego mode, episode runs until timeout
        if ('no_ego' in self.env_config_.keys()) and (self.env_config_['no_ego'] == True):
            done = False
            return False, 0.0, 0.0
        else:
            # 1. r_com: determine whether ego has reached goal
            dist_to_goal = self.__compute_dist_to_observable_state(self.agents_['0'].observable_state_, self.agents_['0'].goal_)
            reached_last_lane = (self.agents_['0'].closest_lane_id_order_ == len(self.agents_['0'].path_)-1)
            if (dist_to_goal < self.done_thres_) and reached_last_lane:
                # reward += 100.0
                conditional_log(self.log_name_, logger, f'Ego {ego_id} reach goal with distance to goal: {dist_to_goal}', 'debug')
                done = True
            # 2. time penalty
            if self.env_config_['reward'] in ['default', 'default_ttc', 'default_fad']:
                w, b = self.env_config_['time_penalty_coeff']
                time_penalty = w * ego_b1 + b
                if ego_id == 0:
                    conditional_log(self.log_name_, logger, f'Ego {ego_id} time penalty {time_penalty:.4f}', 'debug')
                reward += time_penalty
            elif self.env_config_['reward'] == 'simplified':
                reward = reward - 0 # placeholder
            else:
                raise Exception('Invalid reward in env config')
            # 3. r_collision
            for agent_id, agent in self.agents_.items():
                if agent_id == '0':
                    continue
                if self.__detect_collision(self.agents_['0'], agent):
                    w, b = self.env_config_['collision_penalty_coeff']
                    collision_penalty = w * ego_b1 + b
                    if ego_id == 0:
                        conditional_log(self.log_name_, logger, f'ego {ego_id} collision with agent {agent_id}. collision_penalty = {collision_penalty:.4f}', 'debug')
                    reward += collision_penalty
                    done = True
                    collision = True
                    break
            # 4. r_ttc of ego
            if ('reward' in self.env_config_.keys()) and (self.env_config_['reward'] == 'default_ttc'):
                ttc3 = np.min(self.ttc_dp_[3, ego_id, :])
                if ttc3 < self.agents_['0'].ttc_thres_:
                    should_stop = self.agents_['0'].should_stop_wrapped(self.dt_, self.agents_, self.ttc_dp_,
                        self.env_config_['ego_baseline'], self.env_config_['include_agents_within_range'], 
                        self.env_config_['ego_ttc_break_tie'])
                    ego_vel = self.agents_['0'].observable_state_.velocity_
                    if (should_stop and ego_vel > 0.0) or ((not should_stop) and (ego_vel == 0.0)):
                        ttc_penalty = - 5.0 * (self.agents_['0'].ttc_thres_ - ttc3)
                        conditional_log(self.log_name_, logger, f'ego {ego_id} ttc penalty {ttc_penalty:.1f} given ttc3: {ttc3:.1f} | should_stop={should_stop}, ego_vel={ego_vel}', 'debug')
                        reward += ttc_penalty
            else: # just log it (in should_stop), but we don't use it # if use_default_ego, this info is already printed
                if ego_id == 0 and not self.env_config_['use_default_ego']:
                    should_stop = self.agents_['0'].should_stop_wrapped(self.dt_, self.agents_, self.ttc_dp_,
                        self.env_config_['ego_baseline'], self.env_config_['include_agents_within_range'], 
                        self.env_config_['ego_ttc_break_tie'])

            # 5. r_v: velocity reward
            if self.agents_['0'].observable_state_.velocity_ > 0:
                w, b = self.env_config_['velocity_reward_coeff']
                velocity_reward = w * ego_b1 + b
                if ego_id == 0:
                    conditional_log(self.log_name_, logger, f'Ego {ego_id} velocity reward {velocity_reward:.4f}', 'debug')
                reward += velocity_reward
            
            # 6. dist-to-closest-front-agent (fad) penalty 
            if ('reward' in self.env_config_.keys()) and (self.env_config_['reward'] == 'default_fad'): # front-agent-distance
                min_fad, min_fad_agent = self.__calculate_dist_to_closest_front_agent(self.agents_['0'])
                if min_fad_agent != -1:
                    fad_penalty = - 2. + 2. / (1. + math.exp(- (self.env_config_['fad_penalty_coeff']) * min_fad))
                    if ego_id == 0:
                        conditional_log(self.log_name_, logger, f'ego {ego_id} fad penalty {fad_penalty:.3f} given fad: {min_fad:.1f} with agent {min_fad_agent}', 'debug')
                    reward += fad_penalty
            
            # 7. stalemate penalty
            # determine whether ego is on one of the intersection_lanes or its successor or its predecessor
            # if self.__is_ego_near_intersection():
            eps = 1e-5
            temp_matrix = np.tile(self.history_output_parametric_states_[0:1,:,:], (self.env_config_['stalemate_horizon'], 1, 1))
            if (abs(np.sum(self.history_output_parametric_states_ - temp_matrix)) < eps) and \
                (self.curr_action_ == 0):
                w, b = self.env_config_['stalemate_penalty_coeff']
                stalemate_penalty = w * ego_b1 + b
                if ego_id == 0:
                    conditional_log(self.log_name_, logger, f'ego {ego_id} stalemate penalty {stalemate_penalty:.4f} at ts={self.steps_}', 'debug')
                reward += stalemate_penalty
        
            # 8. timeout penalty
            if self.steps_ == self.env_config_['max_episode_timesteps']:
                w, b = self.env_config_['timeout_penalty_coeff']
                timeout_penalty = w * ego_b1 + b
                if ego_id == 0:
                    conditional_log(self.log_name_, logger, f'ego {ego_id} timeout penalty {timeout_penalty:.4f} at ts={self.steps_}', 'debug')
                reward += timeout_penalty
            
            # log total reward
            if ego_id == 0:
                conditional_log(self.log_name_, logger, f'ego {ego_id} reward {reward:.4f}', 'debug')
            return reward, done, collision

    def __is_ego_near_intersection(self):
        ego_lane_id = self.agents_['0'].path_[self.agents_['0'].closest_lane_id_order_]
        intersection_ids = ['t1','t2','t3','t4','r1','r2','r3','r4']
        for intersection_id in intersection_ids:
            for lane_id in intersection_id_dict[intersection_id]:
                if lane_id == ego_lane_id:
                    return True
                else:
                    lane_seg = self.lane_segments_[lane_id]
                    if (ego_lane_id in lane_seg.predecessors) or (ego_lane_id in lane_seg.successors):
                        return True
        return False

    def __compute_dist_to_observable_state(self, 
                                           this_observable_state: ObservableState, 
                                           other_observable_state: ObservableState) -> float:
        # to cover up a bug in datasets
        if isinstance(this_observable_state, Position):
            this_pos = this_observable_state
        else:
            this_pos = this_observable_state.position_
        if isinstance(other_observable_state, Position):
            other_pos = other_observable_state
        else:
            other_pos = other_observable_state.position_

        pos_dist = this_pos.calculate_distance(other_pos)
        # vel_dist = abs(this_observable_state.velocity_ - other_observable_state.velocity_)
        # yaw_rate_dist = abs(this_observable_state.yaw_rate_ - other_observable_state.yaw_rate_)
        return pos_dist
    
    def __calculate_city_frame_to_ego_frame_se3(self) -> SE3:
        ego_position = self.agents_['0'].observable_state_.position_
        city_frame_to_ego_frame_se3 = SE3(rotation=rotation_matrix_z(ego_position.heading_), 
                                          translation=np.array([ego_position.x_, ego_position.y_, 0])).inverse()
        return city_frame_to_ego_frame_se3

    def __detect_collision(self, agent1: Agent, agent2: Agent) -> bool:
        dist = math.sqrt((agent1.observable_state_.position_.x_ - agent2.observable_state_.position_.x_)**2 + \
                     (agent1.observable_state_.position_.y_ - agent2.observable_state_.position_.y_)**2)
        return (dist < agent1.radius_ + agent2.radius_ + 1.0)
    
    def __compute_position_from_lane_waypoint(self, lane_id, waypoint_idx):
        lane_cl = self.lane_segments_[lane_id].centerline
        # supports negative indexing
        if waypoint_idx < 0 and waypoint_idx >= -len(lane_cl):
            waypoint_idx += len(lane_cl)
        # check waypoint_idx within range
        if waypoint_idx >=0 and waypoint_idx < len(lane_cl):
            pos_x = lane_cl[waypoint_idx,0]
            pos_y = lane_cl[waypoint_idx,1]
            if waypoint_idx == len(lane_cl) - 1: # if it's the last one on the lane
                delta_xy = lane_cl[waypoint_idx,:] - lane_cl[waypoint_idx-1,:]
            else:
                delta_xy = lane_cl[waypoint_idx+1,:] - lane_cl[waypoint_idx,:]
            heading = math.atan2(delta_xy[1], delta_xy[0])
            return Position(pos_x, pos_y, heading)
        else:
            raise Exception('Invalid waypoint idx')
    
    def __check_any_collision(self, input_agent: Agent, agents: List[Agent]) -> bool:
        for agent_id, agent in agents.items():
            if agent_id == str(input_agent.id_): # don't compare with self
                continue
            if self.__detect_collision(input_agent, agent):
                return True
        return False

    def __meets_spacing_requirement(self, input_agent: Agent, agents: List[Agent]):
        for other_agent_id, other_agent in agents.items():
            if other_agent_id == str(input_agent.id_): # don't compare with self
                continue
            dist = calculate_poly_distance(input_agent, other_agent)
            # if other agent is in front of me
            if (dist < 1 + self.fad_coeff_ * input_agent.ttc_thres_ * input_agent.v_desired_): # assume 0.5*ttc_thres_ self stop time if the other agent suddenly stops
                other_agent_lane_id = other_agent.path_[other_agent.closest_lane_id_order_]
                other_agent_waypoint_idx = other_agent.closest_waypoint_idx_
                agent_lane_id = input_agent.path_[input_agent.closest_lane_id_order_]
                agent_waypoint_idx = input_agent.closest_waypoint_idx_
                if (other_agent_lane_id == agent_lane_id and agent_waypoint_idx < other_agent_waypoint_idx) or \
                    other_agent_lane_id in (self.lane_segments_[agent_lane_id].successors):
                    return False # doesn't meet requirement
            # now use ttc3
            ttc1 = calculate_time_to_collision(input_agent, other_agent, self.default_ttc_, self.dt_, -1, 1)
            ttc2 = calculate_time_to_collision(input_agent, other_agent, self.default_ttc_, self.dt_, 1, -1)
            ttc3 = calculate_time_to_collision(input_agent, other_agent, self.default_ttc_, self.dt_, 1, 1)
            min_ttc = min(ttc1, min(ttc2, ttc3))
            if min_ttc < input_agent.ttc_thres_ or min_ttc < other_agent.ttc_thres_: # my position has to meet ttc threshold requirement of both me and the other guy
                return False # doesn't meet requirement
        return True

    # == helper functions related to random router ==   
    def __generate_path_in_set(self, lane_id:str, lane_id_set: Set[str]) -> List[str]:
        # generate a path inside a set of lanes
        # it's possible that the starting lane_id itself is not in lane_id_set. 
        queue = deque()
        queue.append(lane_id)
        parents = {lane_id : None}
        path_found = False
        end_id = lane_id

        while not path_found:
            if not queue:
                pdb.set_trace()
            curr_id = queue.popleft()
            successors = copy.deepcopy(self.lane_segments_[curr_id].successors)
            random.shuffle(successors)
            for suc in successors:
                # loop
                if suc not in lane_id_set:
                    path_found = True
                    end_id = curr_id
                else:
                    if suc in list(parents.keys()):
                        continue
                    queue.append(suc)
                    parents[suc] = curr_id

        # trace path
        path = [end_id]
        last_lane_id = parents[end_id]
        while last_lane_id is not None:
            path.append(last_lane_id)
            last_lane_id = parents[last_lane_id]
        path = list(np.flip(path))

        return path

    def __expand_lane_set(self, path: List[str], depth: int) -> List[str]:
        # expand the lanes in given path to a larger neighboring set
        rst_set = set(path)
        # first, add all intersection lanes to the set
        for lane_id in path:
            if self.lane_segments_[lane_id].is_intersection:
                intersection_id = self.lane_segments_[lane_id].intersection_id
                rst_set = rst_set.union(intersection_id_dict[intersection_id])
        rst_set_temp = copy.deepcopy(rst_set)
        # then add all 2-degree pred and suc to the set
        for d in range(depth):
        #   find all pred and suc of path and add to set
            for lane_id in rst_set_temp:
                for pred in self.lane_segments_[lane_id].predecessors:
                    rst_set.add(pred)
                for suc in self.lane_segments_[lane_id].successors:
                    rst_set.add(suc)
            rst_set_temp = copy.deepcopy(rst_set)
        return rst_set

    def generate_agent_extending_path(self, lane_id: str) -> List[str]:
        extending_path_lane_set = set(self.lane_segments_.keys())
        extending_path_lane_set.remove(lane_id)
        # repeat the following extension 10 times so that agent path never runs out
        total_agent_extending_path = []
        for _ in range(10):
            # starting from current lane, plan all the way until a predecessor of current lane
            agent_extending_path = self.__generate_path_in_set(lane_id, extending_path_lane_set) 
            total_agent_extending_path += agent_extending_path
        total_agent_extending_path.append(lane_id) # the path both starts and ends with current lane_id
        return total_agent_extending_path

    def __generate_fixed_depth_path(self, lane_id:str, num_intersections: int) -> List[str]:
        # 1. the path should be a path in intersection
        if not self.lane_segments_[lane_id].is_intersection:
            raise Exception('starting lane_id not in intersection')
        # 2. while curr_num_intersections < num_intersections:
        curr_num_intersections = 1
        aggregate_path = [lane_id]
        while curr_num_intersections < num_intersections:
            # BFS one level deeper with random order into queue
            queue = deque()
            queue.append(lane_id)
            parents = {lane_id : None}
            path_found = False
            end_id = lane_id

            while not path_found:
                if not queue:
                    raise Exception('Can\'t find another intersection?')
                curr_id = queue.popleft()
                successors = copy.deepcopy(self.lane_segments_[curr_id].successors)
                random.shuffle(successors)
                for suc in successors:
                    # loop
                    if self.lane_segments_[suc].is_intersection:
                        path_found = True
                        parents[suc] = curr_id
                        end_id = suc # we want to include suc in our path
                    else:
                        if suc in list(parents.keys()):
                            continue
                        queue.append(suc)
                        parents[suc] = curr_id

            # trace path
            path = [end_id]
            last_lane_id = parents[end_id]
            while last_lane_id is not None:
                path.append(last_lane_id)
                last_lane_id = parents[last_lane_id]
            path = list(np.flip(path))
            # add to aggregate_path
            aggregate_path = aggregate_path + path[1:]
            curr_num_intersections += 1
            # update tracker!
            lane_id = end_id

        # 3. extend one path up and one path down and the resulting path is what we want (lane-intersection*n-lane)
        aggregate_path.insert(0, self.lane_segments_[aggregate_path[0]].predecessors[0] )
        aggregate_path.append(self.lane_segments_[aggregate_path[-1]].successors[0])

        # 3. delete first path and last lane, and the resulting path has num_intersections in it
        # aggregate_path = aggregate_path[1:-1]

        return aggregate_path

    def __initialize_ego(self, num_intersections: int) -> (Position, Position, List[str]):
        # initializes ego state given number of intersections to include in the generated path
        # returns: ego_pos, ego_goal_pos, ego_path

        # 1. pick a random lane in intersection
        if num_intersections == 1:
            if self.env_config_['single_intersection_type'] == 't-intersection':
                intersection_ids = ['t1','t2','t3','t4'] 
                rand_lane_ids = intersection_id_dict[intersection_ids[np.random.randint(len(intersection_ids))]]
            elif self.env_config_['single_intersection_type'] == 'roundabout':
                intersection_ids = ['r1','r2','r3','r4'] 
                rand_lane_ids = intersection_id_dict[intersection_ids[np.random.randint(len(intersection_ids))]]
            elif self.env_config_['single_intersection_type'] == 'turn':
                intersection_ids = ['turn1','turn2','turn3','turn4'] 
                rand_lane_ids = intersection_id_dict[intersection_ids[np.random.randint(len(intersection_ids))]]
            elif self.env_config_['single_intersection_type'] == 'tr':
                intersection_ids = ['t1','t2','t3','t4','r1','r2','r3','r4']
                rand_lane_ids = intersection_id_dict[intersection_ids[np.random.randint(len(intersection_ids))]]
            elif self.env_config_['single_intersection_type'] == 'mix':
                intersection_lane_ids = list(intersection_id_dict.values())
                rand_lane_ids = intersection_lane_ids[np.random.randint(len(intersection_lane_ids))] # lane_ids at a randomly selected intersection
            else:
                raise Exception('Invalid single_intersection_type')
        # if num_intersections > 1, pick whichever intersection type
        else:
            intersection_lane_ids = list(intersection_id_dict.values())
            rand_lane_ids = intersection_lane_ids[np.random.randint(len(intersection_lane_ids))] # lane_ids at a randomly selected intersection
        lane_id = rand_lane_ids[np.random.randint(len(rand_lane_ids))]
        # 2. generate a path from there
        ego_path = self.__generate_fixed_depth_path(lane_id, num_intersections)
        if ('ego_expand_path_depth' in self.env_config_.keys()) and (self.env_config_['ego_expand_path_depth'] > 1):
            for i in range(self.env_config_['ego_expand_path_depth'] - 1):
                # ego_path.insert(0, self.lane_segments_[ego_path[0]].predecessors[0])
                ego_path.append(self.lane_segments_[ego_path[-1]].successors[0])
        # 3. initialize positions based on the path (at the end of last lane)
        # if ('fix_ego_start' not in self.env_config_.keys()) or (self.env_config_['fix_ego_start']):
        #     lane_order = 0
        #     waypt_idx = len(self.lane_segments_[ego_path[lane_order]].centerline) - 250
        # else:
        #     lane_order = np.random.randint(len(ego_path))
        #     lane_cl_len = len(self.lane_segments_[ego_path[lane_order]].centerline)
        #     waypt_idx = np.random.randint(lane_cl_len)
        lane_order = 0
        waypt_idx = len(self.lane_segments_[ego_path[lane_order]].centerline) - 250
        ego_pos = self.__compute_position_from_lane_waypoint(ego_path[lane_order], waypt_idx)
        ego_goal_pos = self.__compute_position_from_lane_waypoint(ego_path[-1], -1) # goal is always at the end of path

        return ego_pos, ego_goal_pos, ego_path, lane_order, waypt_idx

    def __calculate_dist_to_closest_front_agent(self, agent: Agent) -> float:
        # given an agent, give me its distance to its closest front agent
        min_dist = float('Inf')
        closest_front_agent_id = -1

        agent_lane_id = agent.path_[agent.closest_lane_id_order_]
        agent_waypoint_idx = agent.closest_waypoint_idx_
        my_two_degree_successors = get_two_degree_successors(self.lane_segments_, agent_lane_id)

        for other_agent_id, other_agent in self.agents_.items():
            # don't compare with this current agent itself
            if (other_agent_id == agent.id_): 
                continue
            other_agent_lane_id = other_agent.path_[other_agent.closest_lane_id_order_]
            other_agent_waypoint_idx = other_agent.closest_waypoint_idx_
            # determine whether other_agent is in front of agent
            if (other_agent_lane_id == agent_lane_id and agent_waypoint_idx < other_agent_waypoint_idx) or \
                other_agent_lane_id in my_two_degree_successors:
                # calculate polygon distance between the two
                dist = calculate_poly_distance(agent, other_agent)
                if dist < min_dist:
                    min_dist = dist
                    closest_front_agent_id = other_agent_id
        
        return min_dist, closest_front_agent_id
    
    def __assign_agent_models(self, min_id, num_agents) -> List[int]:
        # Note: this method should only be used when no other agent is created from saved agent!
        # if it is used when adding extra agents, then the rl model ratio only applies to the extra agents 

        # model assignment: -1 = default, [0,len(rl_agentconfigs)) = that rl model
        model_assignment = np.ones(min_id + num_agents) * (-1)
        # first check validity 
        cum_sum = 0
        for model_path, model_agent_ratio in self.env_config_['rl_agent_configs']:
            cum_sum += model_agent_ratio
            if cum_sum > 1:
                raise Exception("Invalid ratio in rl_agent_configs")
        # then assign models
        curr_idx = min_id
        for i in range(len(self.env_config_['rl_agent_configs'])):
            model_agent_ratio = self.env_config_['rl_agent_configs'][i][1]
            model_num_agents = int(model_agent_ratio * num_agents) # floor
            model_assignment[curr_idx : (curr_idx + model_num_agents)] = i
            curr_idx += model_num_agents
        return model_assignment.astype(int)
    
    def __find_cycle(self, graph):
        # directed graph: vertex: [(neighbor_vertex, weight)]}
        visited = [False] * (self.env_config_['num_other_agents'] + 1)
        recStack = [False] * (self.env_config_['num_other_agents'] + 1)
        parents = (np.ones(self.env_config_['num_other_agents'] + 1) * (-1)).astype(int)
        for node in range(self.env_config_['num_other_agents'] + 1): 
            if visited[node] == False: 
                is_cycle, cycle_vertex = self.__isCyclicUtil(graph,node,visited,recStack,parents)
                if is_cycle:
                    conditional_log(self.log_name_, logger, f'Cycle detected. parents: {parents}', 'debug')
                    cycle = [cycle_vertex]
                    curr = cycle_vertex
                    while parents[curr] != cycle_vertex:
                        curr = parents[curr]
                        cycle.append(curr)
                    conditional_log(self.log_name_, logger, f'Cycle detected. cycle: {cycle}', 'debug')
                    return True, cycle
        return False, None
    
    def __isCyclicUtil(self, graph, v, visited, recStack, parents): 
        # Mark current node as visited and  
        # adds to recursion stack 
        visited[v] = True
        recStack[v] = True
  
        # Recur for all neighbours 
        # if any neighbour is visited and in  
        # recStack then graph is cyclic 
        for neighbour, _ in graph[v]: 
            if visited[neighbour] == False: 
                parents[neighbour] = v
                is_cycle, cycle_vertex = self.__isCyclicUtil(graph, neighbour, visited, recStack, parents)
                if is_cycle: 
                    return True, cycle_vertex
            elif recStack[neighbour] == True: 
                parents[neighbour] = v # form the cycle
                return True, v # return the end vertex in the cycle
  
        # The node needs to be poped from  
        # recursion stack before function ends 
        recStack[v] = False
        return False, None

    def __generate_current_episode_record(self):
        # return a 1D np array of dim: (4 * (num_other_agents+1),)
        # contains the (x,y,theta,v) of all agents
        
        raw_ep_record = []
        for _, agent in self.agents_.items():
            agent_x = agent.observable_state_.position_.x_
            agent_y = agent.observable_state_.position_.y_
            agent_theta = agent.observable_state_.position_.heading_
            agent_v = agent.observable_state_.velocity_
            raw_ep_record += [agent_x, agent_y, agent_theta, agent_v]
        
        return np.array(raw_ep_record)
    
    def reset_ego_b1(self, b1):
        # helper function to reset all the things that need to be reset in ego
        # in constructor when you change the b1
        self.agents_['0'].saved_agent_.b1_ = b1
        w,b = self.env_config_['ego_velocity_coeff']
        v_desired = w * b1 + b
        self.agents_['0'].saved_agent_.v_desired_ = v_desired
        if (v_desired >= 9):
            self.agents_['0'].saved_agent_.agg_level_ = 2
        elif (v_desired >= 6.8):
            self.agents_['0'].saved_agent_.agg_level_ = 1
        else:
            self.agents_['0'].saved_agent_.agg_level_ = 0
        self.agents_['0'].saved_agent_.observable_state_.velocity_ = v_desired
        self.agents_['0'].initialize_future_horizon_positions()

        ego = self.agents_['0'].saved_agent_

        conditional_log(self.log_name_, logger, f"Reset ego b1: {ego.b1_}, agg_level: {ego.agg_level_}, v_desired_: {ego.v_desired_}",'debug')