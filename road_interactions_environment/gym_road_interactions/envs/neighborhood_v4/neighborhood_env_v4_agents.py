# Closed neighborhood environment with a roundabout, 4 t-intersections

# Python
import pdb, copy, os, sys
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import logging
from queue import Queue
from typing import Any, Dict, Tuple, List
import cv2
import torch
# Argoverse
from argoverse.utils.se3 import SE3
# gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding
# utils
from gym_road_interactions.utils import create_bbox_world_frame, conditional_log, wrap_to_pi
from gym_road_interactions.viz_utils import visualize_agent
from gym_road_interactions.core import AgentType, Position, Agent, ObservableState, Observation, LaneSegment
from shapely.geometry import Point, Polygon
from .neighborhood_env_v4_utils import calculate_remaining_lane_distance, calculate_traversed_lane_distance, calculate_time_to_collision, calculate_poly_distance, get_two_degree_successors

sys.path.append('/home/xiaoyich/masters_thesis_main/policy_network')
from neighborhood_v4_ddqn.models import DDQNAgent, TwinDDQNAgent

logger = logging.getLogger(__name__)

DT = 0.1 # 10Hz
FAD_COEFF = 0.3
TTC_HORIZON = 20
TTC_VEL_PROD = 9.2
DONE_THRES = 0.5 # distance threshold for completing the task

# interface for Neighborhood-v0 agents that creates bbox, v_desired, implements drive_along_path and get_lane_priority and calculate_ttc
class NeighborhoodV4AgentInterface(Agent):
    def __init__(self,
                 id: str,
                 observable_state: ObservableState, 
                 agent_type: AgentType,
                 goal: ObservableState, 
                 curr_lane_id_order: int,
                 curr_waypoint_idx: int,
                 v_desired: float,
                 default_ttc: float,
                 lane_segments: List[LaneSegment],
                 path: list = []): # list of lane_ids

        super(NeighborhoodV4AgentInterface, self).__init__(
                    id=id,
                    observable_state=observable_state, 
                    agent_type=agent_type,
                    goal=goal, 
                    width=2.0, # circle with r = sqrt(2)
                    length=2.0,
                    height=1.7,
                    use_saved_path=True,
                    path=path,
                    closest_lane_id_order=curr_lane_id_order,
                    closest_waypoint_idx=curr_waypoint_idx)

        self.bbox_world_frame_ = create_bbox_world_frame(self) # this function is in gym_road_interactions.utils
        self.v_desired_ = v_desired

        self.radius_ = math.sqrt(2)

        # things related to ttc
        self.default_ttc_ = default_ttc
        self.ttc_thres_ = TTC_VEL_PROD / self.v_desired_
        self.past_ttc_ = [default_ttc, default_ttc, default_ttc]
        self.min_ttc_agent_id_ = ''
        self.ttc_horizon_ = TTC_HORIZON
        self.fad_coeff_ = FAD_COEFF
        self.future_horizon_positions_ = None

        # lane segments
        self.lane_segments_ = lane_segments

        # path_length
        self.path_length_ = self.__calculate_path_length(path)
        self.dist_driven_ = 0

        self.done_thres_ = DONE_THRES
    
    def calculate_ttc(self, agents, dt, ttc_dp) -> None:
        # ttc_dp is a table filled with ttc assuming current speed
        # calculate current-speed ttc with all other agents
        min_ttc = np.min(ttc_dp[0,int(self.id_),:])
        min_ttc_agent_id = np.argmin(ttc_dp[0,int(self.id_),:])

        self.past_ttc_ = self.past_ttc_[1:]
        self.past_ttc_.append(min_ttc)
        self.min_ttc_agent_id_ = min_ttc_agent_id

        # DEBUG
        # conditional_log(self.log_name_, logger,
        #                 f'agent {self.id_} min ttc agent is {min_ttc_agent_id}', 'info')

    def drive_along_path(self, dt: float, assume_vel: int = 0) -> Tuple[Position, int]:
        """
        Returns the end position of this agent if drive along path for dt at current velocity
        assume_vel: 1 (assume at v_desired_), 0 (use current velocity), -1 (assume vel = 0)
        """
        if assume_vel == 1:
            remain_dist = self.v_desired_ * dt
        elif assume_vel == 0:
            remain_dist = self.observable_state_.velocity_ * dt
        elif assume_vel == -1:
            remain_dist = 0
        elif assume_vel == -2: # special mode: drive backwards
            remain_dist = - self.v_desired_ * dt
        else:
            raise Exception(f'Invalid assume_vel value: {assume_vel}')
        
        if remain_dist == 0: # velocity = 0
            # print(f'agent{self.id_} dap: remain_dist <= 0: velocity: {self.observable_state_.velocity_}, remain_dist = {remain_dist}')
            return self.observable_state_.position_, self.closest_lane_id_order_, self.closest_waypoint_idx_
        elif remain_dist > 0: # move forward
            curr_pos = copy.deepcopy(self.observable_state_.position_)
            closest_lane_id_order = self.closest_lane_id_order_

            while remain_dist > 0:
                closest_lane_id_order, remain_dist, curr_pos = self.__drive_along_lane(closest_lane_id_order, remain_dist, curr_pos)
                curr_xy = np.array([[curr_pos.x_, curr_pos.y_]])
                # if self.id_ == '1':
                #     print(f'closest_lane_id_order={closest_lane_id_order}, curr_xy={curr_xy}')
            
            # find closest waypoint index on this updated lane 
            lane_id = self.path_[closest_lane_id_order]
            lane_cl = self.lane_segments_[lane_id].centerline
            curr_xy = np.array([[curr_pos.x_, curr_pos.y_]])
            curr_distances = np.linalg.norm((np.tile(curr_xy, (len(lane_cl), 1)) - lane_cl), axis=1)
            closest_waypoint_idx = np.argmin(curr_distances)
            
            # print(f'agent{self.id_} dap: closest_lane_id_order={closest_lane_id_order}, curr_xy={curr_xy}')
            return curr_pos, closest_lane_id_order, closest_waypoint_idx
        else: # move backward
            curr_pos = copy.deepcopy(self.observable_state_.position_)
            closest_lane_id_order = self.closest_lane_id_order_
            remain_dist = - remain_dist # reverse it so it's easier to debug... just remember that we are driving backwards

            while remain_dist > 0:
                closest_lane_id_order, remain_dist, curr_pos = self.__drive_backwards_along_lane(closest_lane_id_order, remain_dist, curr_pos)
                curr_xy = np.array([[curr_pos.x_, curr_pos.y_]])
                # if self.id_ == '1':
                #     print(f'closest_lane_id_order={closest_lane_id_order}, curr_xy={curr_xy}')
            
            # find closest waypoint index on this updated lane 
            lane_id = self.path_[closest_lane_id_order]
            lane_cl = self.lane_segments_[lane_id].centerline
            curr_xy = np.array([[curr_pos.x_, curr_pos.y_]])
            curr_distances = np.linalg.norm((np.tile(curr_xy, (len(lane_cl), 1)) - lane_cl), axis=1)
            closest_waypoint_idx = np.argmin(curr_distances)
            
            # print(f'agent{self.id_} dap: closest_lane_id_order={closest_lane_id_order}, curr_xy={curr_xy}')
            return curr_pos, closest_lane_id_order, closest_waypoint_idx

    def __drive_along_lane(self, closest_lane_id_order: int, remain_dist: float, curr_pos: Position) -> Tuple[int, float, Position]:
        # returns (closest_lane_id_order, remain_dist, curr_pos)
        # 1. check for zero remain_dist
        if remain_dist == 0:
            return closest_lane_id_order, remain_dist, curr_pos
        # 2. move along current lane
        lane_id = self.path_[closest_lane_id_order]
        curr_lane_seg = self.lane_segments_[lane_id]
        remain_lane_distance = calculate_remaining_lane_distance(lane_id, curr_pos, self.lane_segments_)
        # print(f'remain_lane_distance: {remain_lane_distance}')
        # if remain lane distance is >= remain_dist, we end up on the same lane
        if remain_lane_distance >= remain_dist:
            return_pos = curr_pos
            # straight lanes
            if curr_lane_seg.lane_heading is not None:
                return_pos.x_ = curr_pos.x_ + remain_dist * math.cos(curr_lane_seg.lane_heading)
                return_pos.y_ = curr_pos.y_ + remain_dist * math.sin(curr_lane_seg.lane_heading)
            # curves
            else:
                r = curr_lane_seg.curve_radius
                drive_theta = remain_dist / r
                if curr_lane_seg.turn_direction == 'left': # counter-clockwise
                    return_pos.heading_ = curr_pos.heading_ + drive_theta
                    # theta on circle
                    curr_theta = curr_pos.heading_ - math.pi / 2.0
                else: # clockwise
                    return_pos.heading_ = curr_pos.heading_ - drive_theta
                    # theta on circle
                    curr_theta = curr_pos.heading_ + math.pi / 2.0
                return_pos.x_ = self.lane_segments_[lane_id].curve_center[0] + r * math.cos(curr_theta)
                return_pos.y_ = self.lane_segments_[lane_id].curve_center[1] + r * math.sin(curr_theta)
            return closest_lane_id_order, 0.0, return_pos
        else:
            remain_dist -= remain_lane_distance
            if closest_lane_id_order < len(self.path_) - 1: 
                closest_lane_id_order += 1
                next_lane_id = self.path_[closest_lane_id_order]
                # update curr_pos to be the first waypoint of next lane segment
                next_point = self.lane_segments_[next_lane_id].centerline[0,:]
            else:
                next_lane_id = lane_id
                # we are on the last lane in path. move to the last point on the lane and we are done
                next_point = self.lane_segments_[lane_id].centerline[-1,:]
                remain_dist = 0.0
                
            curr_pos.x_, curr_pos.y_ = next_point[0], next_point[1]
            # correct heading if you come out of a curve (only curves have turn_direction)
            if curr_lane_seg.turn_direction is not None:
                r = curr_lane_seg.curve_radius
                drive_theta = remain_lane_distance / r # remain_lane_distance is the distance that we've driven on this past lane
                if curr_lane_seg.turn_direction == 'right':
                    curr_pos.heading_ = curr_pos.heading_ - drive_theta
                elif curr_lane_seg.turn_direction == 'left':
                    curr_pos.heading_ = curr_pos.heading_ + drive_theta

            curr_pos.heading_ = wrap_to_pi(curr_pos.heading_)
            return closest_lane_id_order, remain_dist, curr_pos

    # remain_dist is positive
    def __drive_backwards_along_lane(self, closest_lane_id_order: int, remain_dist: float, curr_pos: Position) -> Tuple[int, float, Position]:
        # returns (closest_lane_id_order, remain_dist, curr_pos)
        # 1. check for zero remain_dist
        if remain_dist == 0:
            return closest_lane_id_order, remain_dist, curr_pos
        # 2. move backwards along current lane
        lane_id = self.path_[closest_lane_id_order]
        curr_lane_seg = self.lane_segments_[lane_id]
        traversed_lane_distance = calculate_traversed_lane_distance(lane_id, curr_pos, self.lane_segments_)
        # if remain lane distance is >= remain_dist, we end up on the same lane
        if traversed_lane_distance >= remain_dist:
            return_pos = curr_pos
            # straight lanes
            if curr_lane_seg.lane_heading is not None:
                return_pos.x_ = curr_pos.x_ - remain_dist * math.cos(curr_lane_seg.lane_heading)
                return_pos.y_ = curr_pos.y_ - remain_dist * math.sin(curr_lane_seg.lane_heading)
            # curves
            else:
                r = curr_lane_seg.curve_radius
                drive_theta = remain_dist / r
                if curr_lane_seg.turn_direction == 'left': # counter-clockwise
                    return_pos.heading_ = curr_pos.heading_ - drive_theta
                    # theta on circle
                    curr_theta = return_pos.heading_ - math.pi / 2.0
                else: # clockwise
                    return_pos.heading_ = curr_pos.heading_ + drive_theta
                    # theta on circle
                    curr_theta = return_pos.heading_ + math.pi / 2.0
                return_pos.x_ = self.lane_segments_[lane_id].curve_center[0] + r * math.cos(curr_theta)
                return_pos.y_ = self.lane_segments_[lane_id].curve_center[1] + r * math.sin(curr_theta)
            return closest_lane_id_order, 0.0, return_pos
        else:
            remain_dist -= traversed_lane_distance
            if closest_lane_id_order > 0: 
                closest_lane_id_order -= 1
                prev_lane_id = self.path_[closest_lane_id_order]
                # update curr_pos to be the last waypoint of previous lane segment
                prev_point = self.lane_segments_[prev_lane_id].centerline[-1,:]
            else:
                prev_lane_id = lane_id
                # we are on the first lane in path. move to the first point on the lane and we are done
                prev_point = self.lane_segments_[lane_id].centerline[0,:]
                remain_dist = 0.0
                
            curr_pos.x_, curr_pos.y_ = prev_point[0], prev_point[1]
            # correct heading if you come out of a curve (only curves have turn_direction)
            if curr_lane_seg.turn_direction is not None:
                r = curr_lane_seg.curve_radius
                drive_theta = traversed_lane_distance / r # traversed_lane_distance is the distance that we've driven on this past lane
                if curr_lane_seg.turn_direction == 'right':
                    curr_pos.heading_ = curr_pos.heading_ + drive_theta
                elif curr_lane_seg.turn_direction == 'left':
                    curr_pos.heading_ = curr_pos.heading_ - drive_theta

            curr_pos.heading_ = wrap_to_pi(curr_pos.heading_)
            return closest_lane_id_order, remain_dist, curr_pos

    def initialize_future_horizon_positions(self):
        # if v = 0
        curr_x = self.observable_state_.position_.x_
        curr_y = self.observable_state_.position_.y_
        self.future_horizon_positions_ = np.hstack((np.ones((self.ttc_horizon_,1)) * curr_x, \
                                                    np.ones((self.ttc_horizon_,1)) * curr_y)).reshape((1,self.ttc_horizon_,2)) # (1 * ttc_horizon * 2)
        
        # if v > 0
        if len(self.path_) > 0: # if path length > 0, give the actual v > 0 future horizon positions
            temp = []
            for i in range(1, self.ttc_horizon_+1): # 1 to 10
                curr_dt = i * DT
                pred_pos, _, _ = self.drive_along_path(curr_dt, 1)
                temp.append(np.array([[pred_pos.x_, pred_pos.y_]]))
            temp = np.array(temp).reshape((1,self.ttc_horizon_,2))
            self.future_horizon_positions_ = np.concatenate((self.future_horizon_positions_, temp), axis=0) # (2 * ttc_horizon * 2)
        else: # if path length == 0, we want this agent to stay here forever. just repeat v=0 fhp
            self.future_horizon_positions_ = np.concatenate((self.future_horizon_positions_, 
                                                             self.future_horizon_positions_), axis=0) # (2 * ttc_horizon * 2)

    # update future_horizon_positions_ forward by 1 ts
    def update_future_horizon_positions(self):
        if self.observable_state_.velocity_ > 0: # we only need to update fhw if we have moved in this current step
            # if v = 0
            curr_x = self.observable_state_.position_.x_
            curr_y = self.observable_state_.position_.y_
            self.future_horizon_positions_[0,:,:] = np.hstack((np.ones((self.ttc_horizon_,1)) * curr_x, \
                                                            np.ones((self.ttc_horizon_,1)) * curr_y))
            # if v > 0
            pred_pos, _, _ = self.drive_along_path(DT * self.ttc_horizon_, 1)
            next_position = np.array([[pred_pos.x_, pred_pos.y_]])
            self.future_horizon_positions_[1,:,:] = np.vstack((self.future_horizon_positions_[1,1:,:], next_position))

    def __calculate_path_length(self, path):
        tmp_length = 0
        for i in range(len(path)):
            lane_id = path[i]
            tmp_length += self.lane_segments_[lane_id].length
        self.path_length_ = tmp_length

    def apply_action(self, action, dt):
        # action: 0 stop, 1 go, -1 go backwards
        # set velocity
        if action == 0:
            self.observable_state_.velocity_ = 0.0
            self.observable_state_.yaw_rate_ = 0.0
        else:
            self.observable_state_.velocity_ = self.v_desired_
            if action == 1:
                # drive along path with speed
                new_pos, new_closest_lane_id_order, new_closest_point_idx = self.drive_along_path(dt)
            else: # action == -1
                new_pos, new_closest_lane_id_order, new_closest_point_idx = self.drive_along_path(dt, assume_vel=-2)
            self.observable_state_.yaw_rate_ = wrap_to_pi((new_pos.heading_ - self.observable_state_.position_.heading_) / dt)
            # logger.debug(f'[Agent {self.id_}] yaw_rate = {self.observable_state_.yaw_rate_}')
            self.observable_state_.position_ = new_pos
            self.closest_lane_id_order_ = new_closest_lane_id_order
            self.closest_waypoint_idx_ = new_closest_point_idx
            self.dist_driven_ += self.observable_state_.velocity_ * dt
            # logger.info(f'ego action: 1 closest_lane_id_order_={self.closest_lane_id_order_} closest_waypoint_idx_={self.closest_waypoint_idx_}')
            # don't forget to update bbox!!
            self.bbox_world_frame_ = create_bbox_world_frame(self)

    def fad_distance(self):
        return 1 + self.fad_coeff_ * self.ttc_thres_ * self.v_desired_

class NeighborhoodV4DefaultAgent(NeighborhoodV4AgentInterface):
    def __init__(self,
                 id: str,
                 observable_state: ObservableState,
                 goal: ObservableState, 
                 curr_lane_id_order: int,
                 curr_waypoint_idx: int,
                 # agg_level: int, # [0: mild, 1: average, 2: aggressive]
                 default_ttc: float,
                 lane_segments: List[LaneSegment],
                 path: list = [], # list of lane_ids
                 stochastic_stop: bool = False, # whether this agent can choose to stop for ego with a certain prob
                 log_name: str = None,
                 rl_model_path: str = None, # if set, this agent follows a saved RL policy
                 train_configs: dict = None, # used if rl_model_path is not None
                 device = None, # used if rl_model_path is not None
                 v_desired: float = None, # use this if we want to specify agent velocity
                 b1: float = None
                 ): 
        self.log_name_ = log_name

        self.rl_agent_ = None
        self.v_desired_ = v_desired
        self.b1_ = b1

        # == DEPRECATED ==
        # Note: agg_level is determined with b1
        # cf: b1 (driver_type) is a continuous float [-1,1] that determines v_desired_
        # you can view agg_level as a discretization of b1
        # if (v_desired >= 9):
        #     self.agg_level_ = 2
        # elif (v_desired >= 6.8):
        #     self.agg_level_ = 1
        # else:
        #     self.agg_level_ = 0
        # ====
        if (self.b1_ >= 1./3.):
            self.agg_level_ = 2
        elif (v_desired >= -1./3.):
            self.agg_level_ = 1
        else:
            self.agg_level_ = 0

        self.stop_prob_ = - 0.5 * self.b1_ + 0.5

        # if rl_model_path is None:
        #     # choose v_desired and stop_prob based on agg_level
        #     if agg_level == 2:
        #         self.v_desired_ = np.random.uniform(low=9.0, high=11.0) if (v_desired is None) else v_desired
        #     elif agg_level == 1:
        #         self.v_desired_ = np.random.uniform(low=6.8, high=8.8) if (v_desired is None) else v_desired
        #     elif agg_level == 0:
        #         self.v_desired_ = np.random.uniform(low=4.6, high=6.6) if (v_desired is None) else v_desired
        #     else:
        #         raise Exception(f'Invalid agg_level: {agg_level}')
        #     # log
        #     # conditional_log(self.log_name_, logger,
        #     #                 f'agent {id} agg_level: {agg_level}, v_desired_: {self.v_desired_}', 
        #     #                 'debug')
        if rl_model_path is not None:
            # self.v_desired_ = 5.6
            # initialize agent
            self.rl_model_path_ = rl_model_path
            if train_configs['model'] == 'DDQN':
                self.rl_agent_ = DDQNAgent(train_configs, device)
            elif train_configs['model'] == 'TwinDDQN':
                self.rl_agent_ = TwinDDQNAgent(train_configs, device)
            self.rl_agent_.load(self.rl_model_path_)
            self.rl_agent_.value_net.eval()
            # log
            # conditional_log(self.log_name_, logger,
            #                 f'agent {id} is RL agent with policy {rl_model_path}', 
            #                 'debug')

        
        super(NeighborhoodV4DefaultAgent, self).__init__(
                        id=id,
                        observable_state=observable_state, 
                        agent_type=AgentType.other_vehicle, 
                        goal=goal,
                        curr_lane_id_order=curr_lane_id_order,
                        curr_waypoint_idx=curr_waypoint_idx,
                        v_desired = self.v_desired_,
                        default_ttc = default_ttc,
                        lane_segments = lane_segments,
                        path=path)

        self.observable_state_.velocity_ = self.v_desired_
        self.stochastic_stop_ = stochastic_stop
        self.on_stop_ = False # If agent is on stop, it will wait until past_ttc[2] > ttc_thres_

        self.device_ = device
        # track the positions at future 10 ts
        self.initialize_future_horizon_positions()

        self.has_passed_goal_ = False

    def should_stop(self, dt, agents, ttc_dp, ttc_break_tie=None, agent_baseline=None) -> bool:
        # there are 3 scenarios:
        # 1. when following another agent, whoever that is, we will stop when dist <= 1 + 1.5 * self.ttc_thres_ * self.v_desired_
        # 2. otherwise, 
        #   if min_ttc3 < default_ttc (If I go and everyone else goes):
            #   find that agent. ttc1 is the min ttc if I stop, ttc2 is the min ttc if I go.
            #   should_stop = (ttc1 > ttc2) [for ego: should_stop = (ttc >= ttc2)
        # 3. (NOT IMPLEMENTED) if the other agent is ego, and stochastic_stop = True & sample <= stop_prob, return False

        # look up ttc1 (if I stop), ttc2 (if I go)
        # if any of the 4 ttcs are within range, capture that agent
        # (xs, ys)
        ids_ttc_in_range_raw = np.array([]).astype(int)
        for j in range(4):
            # ids_ttc_in_range = np.union1d(ids_ttc_in_range, np.where(ttc_dp[j, int(self.id_), :] < self.default_ttc_)[0])
            ids_ttc_in_range_raw = np.union1d(ids_ttc_in_range_raw, np.where(ttc_dp[j, int(self.id_), :] <= self.ttc_thres_)[0])
        # for crossing vehicles, only consider my interaction with the head of a series of vehicles in ids_ttc_in_range
        ids_ttc_in_range = []
        for raw_id in ids_ttc_in_range_raw:
            to_add = True
            for selected_id in ids_ttc_in_range:
                selected_agent_lane_id = agents[str(selected_id)].path_[agents[str(selected_id)].closest_lane_id_order_]
                selected_agent_waypoint_idx = agents[str(selected_id)].closest_waypoint_idx_
                raw_agent_lane_id = agents[str(raw_id)].path_[agents[str(raw_id)].closest_lane_id_order_]
                raw_agent_waypoint_idx = agents[str(raw_id)].closest_waypoint_idx_
                # if raw agent is behind selected agent, don't add
                if (selected_agent_lane_id == raw_agent_lane_id and raw_agent_waypoint_idx < selected_agent_waypoint_idx) or \
                    selected_agent_lane_id in self.lane_segments_[raw_agent_lane_id].successors:
                    to_add = False
                    break
                # if selected agent is behind raw agent, remove selected agent
                elif (selected_agent_lane_id == raw_agent_lane_id and raw_agent_waypoint_idx > selected_agent_waypoint_idx) or \
                    raw_agent_lane_id in self.lane_segments_[selected_agent_lane_id].successors:
                    ids_ttc_in_range.remove(selected_id)
            if to_add:
                ids_ttc_in_range.append(raw_id)

        ids_should_stop_for = []

        should_stop = False
        my_lane_id = self.path_[self.closest_lane_id_order_]
        my_waypoint_idx = self.closest_waypoint_idx_
        my_two_degree_successors = get_two_degree_successors(self.lane_segments_, my_lane_id)
        for other_agent_id, other_agent in agents.items():
            if (other_agent_id == self.id_): 
                continue
            if len(other_agent.path_) == 0: # special case: other agent path length = 0
                continue
            # if other agent is in front of me
            other_agent_lane_id = other_agent.path_[other_agent.closest_lane_id_order_]
            other_agent_waypoint_idx = other_agent.closest_waypoint_idx_
            dist = calculate_poly_distance(self, other_agent)
            
            # conditional_log(self.log_name_, logger,
            #             f'agent {self.id_} dist to agent {other_agent_id} | dist={dist}', 'info') # DEBUG
             # == front-vehicle logic ==
            if (agent_baseline not in [5]) and \
               ((other_agent_lane_id == my_lane_id and my_waypoint_idx < other_agent_waypoint_idx) or \
                (other_agent_lane_id in my_two_degree_successors)):
                if (dist <= 1 + self.fad_coeff_ * self.ttc_thres_ * self.v_desired_): # assume ttc_thres_ self stop time if the other agent suddenly stops
                    # conditional_log(self.log_name_, logger,
                    #     f'agent {self.id_} agent_baseline={agent_baseline} | should stop for front agent {other_agent_id} | dist={dist}', 'debug') # DEBUG
                    est_decision_ttc = dist / self.v_desired_
                    ids_should_stop_for.append((int(other_agent_id), est_decision_ttc))
                    should_stop = (should_stop or True) # we want to go through everyone for debug
            else:
                i = int(other_agent_id)
                if i in ids_ttc_in_range:
                    # == back-vehicle logic ==
                    # if this ttc3 agent is behind me, don't consider whether we should stop for them
                    if (agent_baseline not in [5]) and \
                       ((my_lane_id == other_agent_lane_id and other_agent_waypoint_idx < my_waypoint_idx) or \
                        my_lane_id in self.lane_segments_[other_agent_lane_id].successors):
                        # conditional_log(self.log_name_, logger, f'agent {self.id_} agent_baseline={agent_baseline} | with agent {i} behind. shouldn\'t stop.', 'debug')
                        should_stop = (should_stop or False)
                    # == ttc logic ==
                    else: 
                        ttc3 = ttc_dp[3, int(self.id_), i] # if we both go
                        ttc1 = ttc_dp[1, int(self.id_), i] # if I stop
                        ttc2 = ttc_dp[2, int(self.id_), i] # if I go
                        # max_ttc_go = max(ttc2, ttc3)
                        if ttc1 == ttc2:
                            # based on who's agg_level = 0
                            if ttc_break_tie is None or ttc_break_tie == 'agg_level=0':
                                # if I'm mild, the other is not, I should stop
                                if self.agg_level_ == 0 and other_agent.agg_level_ != 0:
                                    criteria = True 
                                # if both are mild, break tie: larger id stops (this is to make sure agent stops for ego)
                                elif self.agg_level_ == 0 and other_agent.agg_level_ == 0:
                                    criteria = (int(self.id_) > i) 
                                # if I'm not mild and the other is mild, I shouldn't stop
                                elif self.agg_level_ != 0 and other_agent.agg_level_ == 0:
                                    criteria = False
                                # if both are not mild, break tie: larger id stops
                                elif self.agg_level_ != 0 and other_agent.agg_level_ != 0:
                                    criteria = (int(self.id_) > i) 
                                # conditional_log(self.log_name_, logger, 
                                #     f'agent {self.id_} agent_baseline={agent_baseline} | ttc_break_tie: {ttc_break_tie} | criteria={criteria}', 'debug')
                            # based on b1
                            elif ttc_break_tie == 'b1':
                                if self.b1_ != other_agent.b1_:
                                    criteria = (self.b1_ < other_agent.b1_)
                                else:
                                    temp = np.random.random()
                                    if temp < 0.5:
                                        criteria = True
                                    else:
                                        criteria = False
                                # conditional_log(self.log_name_, logger, 
                                #     f'agent {self.id_} agent_baseline={agent_baseline} | ttc_break_tie: {ttc_break_tie} | criteria={criteria}', 'debug')
                            # based on coin flip
                            elif ttc_break_tie == 'random':
                                ttc_sample = np.random.random() # [0,1)
                                criteria = (ttc_sample < self.stop_prob_)
                                # conditional_log(self.log_name_, logger,
                                #     f'agent {self.id_} agent_baseline={agent_baseline} | ttc_break_tie: {ttc_break_tie} | sample={ttc_sample} | stop_prob = {self.stop_prob_} | criteria={criteria}', 'debug') 
                            else:
                                raise Exception(f"Invalid ttc_break_tie: {ttc_break_tie}")
                        else:
                            criteria = (ttc1 > ttc2)
                            # conditional_log(self.log_name_, logger,
                            #         f'agent {self.id_} agent_baseline={agent_baseline} | ttc | criteria={criteria}', 'debug') 
                        # debug
                        if criteria:
                            ids_should_stop_for.append((i, ttc2))
                        
                        should_stop = (should_stop or criteria) # we could break loop here, but calculate all so we can debug
                        # conditional_log(self.log_name_, logger,
                        #             f'agent {self.id_} (agg={self.agg_level_}) ttc with agent {i} (agg={other_agent.agg_level_}) | ttc3 (both go) = {ttc3} | ttc1 (I stop) = {ttc1} | ttc2 (I go) = {ttc2} | should stop: {criteria}', 'debug') 

        # if should_stop:
        #     conditional_log(self.log_name_, logger,
        #                 f'agent {self.id_} should stop for: {str(ids_should_stop_for)}', 'debug') 
    
        return should_stop, ids_should_stop_for

    def step(self, dt, agents, ttc_dp, agent_states, agent_action_noise=0, ttc_break_tie=None, agent_baseline=None) -> None:
        should_stop, ids_should_stop_for = self.should_stop(dt, agents, ttc_dp, ttc_break_tie, agent_baseline)
        action = 0
        # if not rl agent, plan as usual
        if self.rl_agent_ is None:
            # conditional_log(self.log_name_, logger,
            #                 f'agent {self.id_} ttc: {str(self.past_ttc_)}', 'info') # DEBUG
            # first checking whether it's possible to switch out of on_stop_
            if self.on_stop_ and (not should_stop):
                # conditional_log(self.log_name_, logger,
                #             f'agent {self.id_} leaving on_stop_', 'debug')
                self.on_stop_ = False

            if not self.on_stop_:
                # decide whether to enter on_stop_
                if should_stop:
                    # stop
                    # conditional_log(self.log_name_, logger,
                    #         f'agent {self.id_} entering on_stop_. action: 0', 'debug')
                    action = 0
                    self.on_stop_ = True
                else:
                    action = 1
        # if rl agent, use rl agent to select action
        else:  
            parametric_state = agent_states[int(self.id_) - 1]
            parametric_state_ts = torch.from_numpy(parametric_state).unsqueeze(0).float().to(self.device_) # 1*(state_dim)
            rl_action = self.rl_agent_.select_action(parametric_state_ts, 0, test=True)
            action = rl_action

        # agent_action_noise
        if agent_action_noise > 0:
            agent_action_noise_sample = np.random.random()
            if (agent_action_noise_sample <= agent_action_noise):
                action = int(1 - action)
                conditional_log(self.log_name_, logger,
                        f'agent {self.id_} switch action to {action} given agent_action_noise={agent_action_noise}', 'debug') # DEBUG

        self.apply_action(action, dt)

        # update self.future_horizon_positions_
        self.update_future_horizon_positions()

        # update whether I have passed goal
        # dist_to_goal = self.__compute_dist_to_observable_state(self.observable_state_, self.goal_)
        # if (dist_to_goal < self.done_thres_):
        #     conditional_log(self.log_name_, logger, f'Agent {self.id_} passed goal with distance to goal: {dist_to_goal}', 'debug')
        #     self.has_passed_goal_ = True

        return ids_should_stop_for
    
    def force_action(self, action, dt):
        # force the agent to take a certain action. Only used for break stop_for cycles or creating test set using backwards driving
        self.apply_action(action, dt)
        # update self.future_horizon_positions_
        self.update_future_horizon_positions()
    
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

# Wrapper around NeighborhoodV4DefaultAgent and NeighborhoodV3DefaultAgent objects that were saved in interaction
# and collision sets in order to modify agent should_stop behavior
class NeighborhoodV4DefaultAgentWrapper(object):
    def __init__(self,
                 saved_agent: NeighborhoodV4DefaultAgent): 
        self.saved_agent_ = copy.deepcopy(saved_agent)
        self.__update_data_members()
    
    def calculate_ttc(self, agents, dt, ttc_dp) -> None:
        rst = self.saved_agent_.calculate_ttc(agents, dt, ttc_dp)
        self.__update_data_members()
        return rst
    
    def drive_along_path(self, dt: float, assume_vel: int = 0) -> Tuple[Position, int]:
        rst = self.saved_agent_.drive_along_path(dt, assume_vel)
        self.__update_data_members()
        return rst

    def initialize_future_horizon_positions(self):
        self.saved_agent_.initialize_future_horizon_positions()
        self.__update_data_members()
    
    def update_future_horizon_positions(self):
        self.saved_agent_.update_future_horizon_positions()
        self.__update_data_members()
    
    def apply_action(self, action, dt):
        self.saved_agent_.apply_action(action, dt)
        self.__update_data_members()
    
    def fad_distance(self):
        return self.saved_agent_.fad_distance()

    def should_stop_wrapped(self, dt, agents, ttc_dp, ttc_break_tie=None, agent_baseline=None) -> bool:
        # look up ttc1 (if I stop), ttc2 (if I go)
        # if any of the 4 ttcs are within range, capture that agent
        # (xs, ys)
        ids_ttc_in_range_raw = np.array([]).astype(int)
        for j in range(4):
            # ids_ttc_in_range = np.union1d(ids_ttc_in_range, np.where(ttc_dp[j, int(self.id_), :] < self.default_ttc_)[0])
            ids_ttc_in_range_raw = np.union1d(ids_ttc_in_range_raw, np.where(ttc_dp[j, int(self.id_), :] <= self.ttc_thres_)[0])
        # for crossing vehicles, only consider my interaction with the head of a series of vehicles in ids_ttc_in_range
        ids_ttc_in_range = []
        for raw_id in ids_ttc_in_range_raw:
            to_add = True
            for selected_id in ids_ttc_in_range:
                selected_agent_lane_id = agents[str(selected_id)].path_[agents[str(selected_id)].closest_lane_id_order_]
                selected_agent_waypoint_idx = agents[str(selected_id)].closest_waypoint_idx_
                raw_agent_lane_id = agents[str(raw_id)].path_[agents[str(raw_id)].closest_lane_id_order_]
                raw_agent_waypoint_idx = agents[str(raw_id)].closest_waypoint_idx_
                # if raw agent is behind selected agent, don't add
                if (selected_agent_lane_id == raw_agent_lane_id and raw_agent_waypoint_idx < selected_agent_waypoint_idx) or \
                    selected_agent_lane_id in self.lane_segments_[raw_agent_lane_id].successors:
                    to_add = False
                    break
                # if selected agent is behind raw agent, remove selected agent
                elif (selected_agent_lane_id == raw_agent_lane_id and raw_agent_waypoint_idx > selected_agent_waypoint_idx) or \
                    raw_agent_lane_id in self.lane_segments_[selected_agent_lane_id].successors:
                    ids_ttc_in_range.remove(selected_id)
            if to_add:
                ids_ttc_in_range.append(raw_id)

        ids_should_stop_for = []

        should_stop = False
        my_lane_id = self.path_[self.closest_lane_id_order_]
        my_waypoint_idx = self.closest_waypoint_idx_
        my_two_degree_successors = get_two_degree_successors(self.lane_segments_, my_lane_id)
        for other_agent_id, other_agent in agents.items():
            if (other_agent_id == self.id_): 
                continue
            if len(other_agent.path_) == 0: # special case: other agent path length = 0
                continue
            
            # == front-vehicle logic ==
            # if other agent is in front of me
            other_agent_lane_id = other_agent.path_[other_agent.closest_lane_id_order_]
            other_agent_waypoint_idx = other_agent.closest_waypoint_idx_
            dist = calculate_poly_distance(self, other_agent)
            # conditional_log(self.log_name_, logger,
            #             f'agent {self.id_} dist to agent {other_agent_id} | dist={dist}', 'info') # DEBUG
            if (agent_baseline not in [5]) and \
               ((other_agent_lane_id == my_lane_id and my_waypoint_idx < other_agent_waypoint_idx) or \
                (other_agent_lane_id in my_two_degree_successors)):
                if (dist <= 1 + self.fad_coeff_ * self.ttc_thres_ * self.v_desired_): # assume ttc_thres_ self stop time if the other agent suddenly stops
                    # conditional_log(self.log_name_, logger,
                    #     f'agent {self.id_} agent_baseline={agent_baseline} | should stop for front agent {other_agent_id} | dist={dist}', 'debug') # DEBUG
                    est_decision_ttc = dist / self.v_desired_
                    ids_should_stop_for.append((int(other_agent_id), est_decision_ttc))
                    should_stop = (should_stop or True) # we want to go through everyone for debug
            else:
                i = int(other_agent_id)
                if i in ids_ttc_in_range:
                    # == back-vehicle logic ==
                    # if this ttc3 agent is behind me, don't consider whether we should stop for them
                    if (agent_baseline not in [5]) and \
                       ((my_lane_id == other_agent_lane_id and other_agent_waypoint_idx < my_waypoint_idx) or \
                        my_lane_id in self.lane_segments_[other_agent_lane_id].successors):
                        # conditional_log(self.log_name_, logger, f'agent {self.id_} agent_baseline={agent_baseline} | with agent {i} behind. shouldn\'t stop.', 'debug')
                        should_stop = (should_stop or False)
                    # == ttc logic ==
                    elif (agent_baseline not in [6]):
                        ttc3 = ttc_dp[3, int(self.id_), i] # if we both go
                        ttc1 = ttc_dp[1, int(self.id_), i] # if I stop
                        ttc2 = ttc_dp[2, int(self.id_), i] # if I go
                        # max_ttc_go = max(ttc2, ttc3)
                        if ttc1 == ttc2:
                            # based on who's agg_level = 0
                            if ttc_break_tie is None or ttc_break_tie == 'agg_level=0':
                                # if I'm mild, the other is not, I should stop
                                if self.agg_level_ == 0 and other_agent.agg_level_ != 0:
                                    criteria = True 
                                # if both are mild, break tie: larger id stops (this is to make sure agent stops for ego)
                                elif self.agg_level_ == 0 and other_agent.agg_level_ == 0:
                                    criteria = (int(self.id_) > i) 
                                # if I'm not mild and the other is mild, I shouldn't stop
                                elif self.agg_level_ != 0 and other_agent.agg_level_ == 0:
                                    criteria = False
                                # if both are not mild, break tie: larger id stops
                                elif self.agg_level_ != 0 and other_agent.agg_level_ != 0:
                                    criteria = (int(self.id_) > i) 
                                # conditional_log(self.log_name_, logger, 
                                #     f'agent {self.id_} agent_baseline={agent_baseline} | ttc_break_tie: {ttc_break_tie} | criteria={criteria}', 'debug')
                            # based on b1
                            elif ttc_break_tie == 'b1':
                                if self.b1_ != other_agent.b1_:
                                    criteria = (self.b1_ < other_agent.b1_)
                                else:
                                    temp = np.random.random()
                                    if temp < 0.5:
                                        criteria = True
                                    else:
                                        criteria = False
                                # conditional_log(self.log_name_, logger, 
                                #     f'agent {self.id_} agent_baseline={agent_baseline} | ttc_break_tie: {ttc_break_tie} | criteria={criteria}', 'debug')
                            # based on coin flip
                            elif ttc_break_tie == 'random':
                                ttc_sample = np.random.random() # [0,1)
                                criteria = (ttc_sample < self.stop_prob_)
                                # conditional_log(self.log_name_, logger,
                                #     f'agent {self.id_} agent_baseline={agent_baseline} | ttc_break_tie: {ttc_break_tie} | sample={ttc_sample} | stop_prob = {self.stop_prob_} | criteria={criteria}', 'debug') 
                            elif ttc_break_tie == 'id':
                                # between agents: larger id go
                                if (other_agent.id_ != '0'):
                                    criteria = (int(self.id_) < int(other_agent.id_))
                                # between agent and ego: larger pseudo_id go
                                else:
                                    criteria = (int(self.id_) < int(other_agent.pseudo_id_))
                                log_str = f'agent {self.id_} other_agent {other_agent.id_} agent_baseline={agent_baseline} | ttc_break_tie: {ttc_break_tie} | criteria={criteria}'
                                if other_agent.id_ == '0':
                                    log_str += f' | other_agent pseudo_id: {other_agent.pseudo_id_}'
                                conditional_log(self.log_name_, logger, log_str, 'debug') 
                            else:
                                raise Exception(f"Invalid ttc_break_tie: {ttc_break_tie}")
                        else:
                            criteria = (ttc1 > ttc2)
                            # conditional_log(self.log_name_, logger,
                            #         f'agent {self.id_} agent_baseline={agent_baseline} | ttc | criteria={criteria}', 'debug') 
                        # debug
                        if criteria:
                            ids_should_stop_for.append((i, ttc2))
                        
                        should_stop = (should_stop or criteria) # we could break loop here, but calculate all so we can debug
                        # conditional_log(self.log_name_, logger,
                        #             f'agent {self.id_} (agg={self.agg_level_}) ttc with agent {i} (agg={other_agent.agg_level_}) | ttc3 (both go) = {ttc3} | ttc1 (I stop) = {ttc1} | ttc2 (I go) = {ttc2} | should stop: {criteria}', 'debug') 

        # if should_stop:
        #     conditional_log(self.log_name_, logger,
        #                 f'agent {self.id_} should stop for: {str(ids_should_stop_for)}', 'debug') 

        self.__update_data_members()

        return should_stop, ids_should_stop_for

    def step(self, dt, agents, ttc_dp, agent_states, agent_action_noise=0, ttc_break_tie=None, agent_baseline=None) -> None:
        should_stop, ids_should_stop_for = self.should_stop_wrapped(dt, agents, ttc_dp, ttc_break_tie, agent_baseline)
        action = 0
        # if not rl agent, plan as usual
        if self.rl_agent_ is None:
            # conditional_log(self.log_name_, logger,
            #                 f'agent {self.id_} ttc: {str(self.past_ttc_)}', 'info') # DEBUG
            # first checking whether it's possible to switch out of on_stop_
            if self.on_stop_ and (not should_stop):
                # conditional_log(self.log_name_, logger,
                #             f'agent {self.id_} leaving on_stop_', 'debug')
                self.on_stop_ = False

            if not self.on_stop_:
                # decide whether to enter on_stop_
                if should_stop:
                    # stop
                    # conditional_log(self.log_name_, logger,
                    #         f'agent {self.id_} entering on_stop_. action: 0', 'debug')
                    action = 0
                    self.on_stop_ = True
                else:
                    action = 1
        # if rl agent, use rl agent to select action
        else:  
            parametric_state = agent_states[int(self.id_) - 1]
            parametric_state_ts = torch.from_numpy(parametric_state).unsqueeze(0).float().to(self.device_) # 1*(state_dim)
            rl_action = self.rl_agent_.select_action(parametric_state_ts, 0, test=True)
            action = rl_action

        # agent_action_noise
        if agent_action_noise > 0:
            agent_action_noise_sample = np.random.random()
            if (agent_action_noise_sample <= agent_action_noise):
                action = int(1 - action)
                conditional_log(self.log_name_, logger,
                        f'agent {self.id_} switch action to {action} given agent_action_noise={agent_action_noise}', 'debug') # DEBUG

        self.apply_action(action, dt)

        # update self.future_horizon_positions_
        self.update_future_horizon_positions()

        # update whether I have passed goal
        # dist_to_goal = self.__compute_dist_to_observable_state(self.observable_state_, self.goal_)
        # if (dist_to_goal < self.done_thres_):
        #     conditional_log(self.log_name_, logger, f'Agent {self.id_} passed goal with distance to goal: {dist_to_goal}', 'debug')
        #     self.has_passed_goal_ = True

        self.__update_data_members()

        return ids_should_stop_for
    
    def force_action(self, action, dt):
        self.saved_agent_.force_action(action, dt)
        self.__update_data_members()
    
    def __compute_dist_to_observable_state(self, 
                                           this_observable_state: ObservableState, 
                                           other_observable_state: ObservableState) -> float:
        rst = self.saved_agent_.__compute_dist_to_observable_state(this_observable_state, other_observable_state)
        self.__update_data_members()
        return rst
    
    def __update_data_members(self):
        self.__dict__.update(self.saved_agent_.__dict__)

class NeighborhoodV4EgoAgent(NeighborhoodV4AgentInterface):
    def __init__(self,
                 id: str,
                 observable_state: ObservableState, 
                 goal: ObservableState, 
                 curr_lane_id_order: int,
                 curr_waypoint_idx: int,
                 default_ttc: float,
                 lane_segments: List[LaneSegment],
                 path: list = [], # list of lane_ids
                 log_name: str = None,
                 v_desired: float = 5.6,
                 b1: float = -1.0): 
        
        self.v_desired_ = v_desired
        # driver_type
        self.b1_ = b1 
        # Note: agg_level is determined with the same velocity range as default agents, {0,1,2}
        # cf: b1 (driver_type) is a continuous float [-1,1] that determines v_desired_
        # you can view agg_level as a discretization of b1
        if (v_desired >= 9):
            self.agg_level_ = 2
        elif (v_desired >= 6.8):
            self.agg_level_ = 1
        else:
            self.agg_level_ = 0
        self.stop_prob_ = - 0.5 * self.b1_ + 0.5

        super(NeighborhoodV4EgoAgent, self).__init__(
                        id=id,
                        observable_state=observable_state, 
                        agent_type=AgentType.ego_vehicle, 
                        goal=goal,
                        curr_lane_id_order=curr_lane_id_order,
                        curr_waypoint_idx=curr_waypoint_idx,
                        v_desired = self.v_desired_,
                        default_ttc = default_ttc,
                        lane_segments = lane_segments,
                        path=path)
        
        self.observable_state_.velocity_ = self.v_desired_
        self.log_name_ = log_name

        # track the waypoints at future 10 ts
        self.initialize_future_horizon_positions()
        
    def step(self, action: int, dt: float) -> None:
        self.apply_action(action, dt)

        # update self.future_horizon_positions_
        self.update_future_horizon_positions()

    def should_stop(self, dt, agents, ttc_dp, 
        ego_baseline=None, include_agents_within_range=None, ttc_break_tie=None) -> bool:
        # look up ttc1 (if I stop), ttc2 (if I go)
        # if any of the 4 ttcs are within range, capture that agent
        # (xs, ys)
        ids_ttc_in_range_raw = np.array([]).astype(int)
        for j in range(4):
            # ids_ttc_in_range = np.union1d(ids_ttc_in_range, np.where(ttc_dp[j, int(self.id_), :] < self.default_ttc_)[0])
            ids_ttc_in_range_raw = np.union1d(ids_ttc_in_range_raw, np.where(ttc_dp[j, int(self.id_), :] <= self.ttc_thres_)[0])
        # if we enforce the same observation range as the model
        if include_agents_within_range is not None and include_agents_within_range > 0:
            indices_to_delete = []
            ids_to_delete = [] # for debug
            for i in range(len(ids_ttc_in_range_raw)):
                agent_id = str(ids_ttc_in_range_raw[i])
                poly_dist = calculate_poly_distance(agents['0'], agents[agent_id])
                if (poly_dist > include_agents_within_range):
                    indices_to_delete.append(i)
                    ids_to_delete.append(agent_id)
            ids_ttc_in_range_raw = np.delete(ids_ttc_in_range_raw, indices_to_delete)
            conditional_log(self.log_name_, logger, f'ttc_in_range ids not in distance range: {str(ids_to_delete)} | ids_ttc_in_range_raw={str(ids_ttc_in_range_raw)}', 'debug')
        # for crossing vehicles, only consider my interaction with the head of a series of vehicles in ids_ttc_in_range
        ids_ttc_in_range = []
        for raw_id in ids_ttc_in_range_raw:
            to_add = True
            for selected_id in ids_ttc_in_range:
                selected_agent_lane_id = agents[str(selected_id)].path_[agents[str(selected_id)].closest_lane_id_order_]
                selected_agent_waypoint_idx = agents[str(selected_id)].closest_waypoint_idx_
                raw_agent_lane_id = agents[str(raw_id)].path_[agents[str(raw_id)].closest_lane_id_order_]
                raw_agent_waypoint_idx = agents[str(raw_id)].closest_waypoint_idx_
                # if raw agent is behind selected agent, don't add
                if (selected_agent_lane_id == raw_agent_lane_id and raw_agent_waypoint_idx < selected_agent_waypoint_idx) or \
                    selected_agent_lane_id in self.lane_segments_[raw_agent_lane_id].successors:
                    to_add = False
                    break
                # if selected agent is behind raw agent, remove selected agent
                elif (selected_agent_lane_id == raw_agent_lane_id and raw_agent_waypoint_idx > selected_agent_waypoint_idx) or \
                    raw_agent_lane_id in self.lane_segments_[selected_agent_lane_id].successors:
                    ids_ttc_in_range.remove(selected_id)
            if to_add:
                ids_ttc_in_range.append(raw_id)

        ids_should_stop_for = []

        should_stop = False
        ego_lane_id = self.path_[self.closest_lane_id_order_]
        ego_waypoint_idx = self.closest_waypoint_idx_
        my_two_degree_successors = get_two_degree_successors(self.lane_segments_, ego_lane_id)
        for other_agent_id, other_agent in agents.items():
            if (other_agent_id == self.id_): 
                continue
            # if other agent is in front of me
            other_agent_lane_id = other_agent.path_[other_agent.closest_lane_id_order_]
            other_agent_waypoint_idx = other_agent.closest_waypoint_idx_
            dist = calculate_poly_distance(self, other_agent)
            # conditional_log(self.log_name_, logger,
            #             f'agent {self.id_} dist to agent {other_agent_id} | dist={dist}', 'info') # DEBUG
            # == front-vehicle logic ==
            if (ego_baseline not in [1,2,5]) and \
                ((other_agent_lane_id == ego_lane_id and ego_waypoint_idx < other_agent_waypoint_idx) or \
                other_agent_lane_id in my_two_degree_successors):
                if (dist <= 1 + self.fad_coeff_ * self.ttc_thres_ * self.v_desired_): # assume ttc_thres_ self stop time if the other agent suddenly stops
                    conditional_log(self.log_name_, logger,
                        f'ego should stop for front agent {other_agent_id} | dist={dist}', 'debug') # DEBUG
                    est_decision_ttc = dist / self.v_desired_
                    ids_should_stop_for.append((int(other_agent_id), est_decision_ttc))
                    should_stop = (should_stop or True) # we want to go through everyone for debug
            else:
                i = int(other_agent_id)
                if i in ids_ttc_in_range:
                    # == back-vehicle logic ==
                    # if this ttc3 agent is behind me, don't consider whether we should stop for them
                    if  (ego_baseline not in [5]) and \
                        ((ego_lane_id == other_agent_lane_id and other_agent_waypoint_idx < ego_waypoint_idx) or \
                         (ego_lane_id in self.lane_segments_[other_agent_lane_id].successors)):
                        conditional_log(self.log_name_, logger, f'ego with agent {i} behind. shouldn\'t stop.', 'debug')
                        should_stop = (should_stop or False)
                    # == ttc logic ==
                    elif (ego_baseline not in [6]):
                        ttc3 = ttc_dp[3, int(self.id_), i] # if we both go
                        ttc1 = ttc_dp[1, int(self.id_), i] # if I stop
                        ttc2 = ttc_dp[2, int(self.id_), i] # if I go
                        # max_ttc_go = max(ttc2, ttc3)
                        # criteria = (ttc1 >= max_ttc_go)
                        if ttc1 == ttc2:
                            if ego_baseline in [None, 5, 4]:
                                # based on who's agg_level = 0
                                if (ego_baseline not in [4]) and (ttc_break_tie is None or ttc_break_tie == 'agg_level=0'):
                                    # if the other is not mild, I should stop
                                    if other_agent.agg_level_ != 0:
                                        criteria = True 
                                    # if the other is mild, I should go
                                    else:
                                        criteria = False
                                    conditional_log(self.log_name_, logger, 
                                        f'agent {self.id_} ttc_break_tie: {ttc_break_tie} | criteria={criteria}', 'debug')
                                # based on b1
                                elif (ego_baseline not in [4]) and (ttc_break_tie == 'b1'):
                                    conditional_log(self.log_name_, logger, f'agent {self.id_} ttc_break_tie: {ttc_break_tie}', 'debug')
                                    if self.b1_ != other_agent.b1_:
                                        criteria = (self.b1_ < other_agent.b1_)
                                    else:
                                        temp = np.random.random()
                                        if temp < 0.5:
                                            criteria = True
                                        else:
                                            criteria = False
                                    conditional_log(self.log_name_, logger, 
                                        f'agent {self.id_} ttc_break_tie: {ttc_break_tie} | criteria={criteria}', 'debug')
                                # based on coin flip
                                elif ttc_break_tie == 'random':
                                    ttc_sample = np.random.random() # [0,1)
                                    criteria = (ttc_sample < self.stop_prob_)
                                    conditional_log(self.log_name_, logger,
                                        f'agent {self.id_} ttc_break_tie: {ttc_break_tie} | sample={ttc_sample} | stop_prob = {self.stop_prob_} | criteria={criteria}', 'debug') 
                                else:
                                    raise Exception(f"invalid ttc_break_tie: {ttc_break_tie}")
                            else:
                                if ego_baseline in [0,1]:
                                    temp = np.random.random()
                                    if temp < 0.5:
                                        criteria = True
                                    else:
                                        criteria = False
                                else:
                                    raise Exception(f"invalid ego_baseline: {ego_baseline}")
                        else:
                            criteria = (ttc1 > ttc2)
                        # debug
                        if criteria:
                            ids_should_stop_for.append((i, ttc2))
                        
                        should_stop = (should_stop or criteria) # we could break loop here, but calculate all so we can debug
                        conditional_log(self.log_name_, logger,
                                    f'agent {self.id_} (agg={self.agg_level_}) ttc with agent {i} (agg={other_agent.agg_level_}) | ttc3 (both go) = {ttc3} | ttc1 (I stop) = {ttc1} | ttc2 (I go) = {ttc2}', 'debug') 
                        if ego_baseline is None:
                            conditional_log(self.log_name_, logger,
                                    f'agent {self.id_} (agg={self.agg_level_}) with agent {i} | ttc logic | should stop: {criteria}', 'debug') 
                        else:
                            conditional_log(self.log_name_, logger,
                                    f'agent {self.id_} (agg={self.agg_level_}) with agent {i} | ttc logic | ego_baseline {ego_baseline} should_stop: {criteria}', 'debug') 
                    
        if should_stop:
            conditional_log(self.log_name_, logger,
                        f'agent {self.id_} should stop for: {str(ids_should_stop_for)}', 'debug') 
    
        return should_stop, ids_should_stop_for


class NeighborhoodV4EgoAgentWrapper(NeighborhoodV4AgentInterface):
    def __init__(self,
                 saved_agent: NeighborhoodV4EgoAgent,
                 max_num_other_agents: int): 
        self.saved_agent_ = copy.deepcopy(saved_agent)
        self.pseudo_id_ = str(np.random.randint(max_num_other_agents + 1))
        self.__update_data_members()
    
    def calculate_ttc(self, agents, dt, ttc_dp) -> None:
        rst = self.saved_agent_.calculate_ttc(agents, dt, ttc_dp)
        self.__update_data_members()
        return rst
    
    def drive_along_path(self, dt: float, assume_vel: int = 0) -> Tuple[Position, int]:
        rst = self.saved_agent_.drive_along_path(dt, assume_vel)
        self.__update_data_members()
        return rst

    def initialize_future_horizon_positions(self):
        self.saved_agent_.initialize_future_horizon_positions()
        self.__update_data_members()
    
    def update_future_horizon_positions(self):
        self.saved_agent_.update_future_horizon_positions()
        self.__update_data_members()
    
    def apply_action(self, action, dt):
        self.saved_agent_.apply_action(action, dt)
        self.__update_data_members()
    
    def fad_distance(self):
        return self.saved_agent_.fad_distance()

    def step(self, action: int, dt: float) -> None:
        self.saved_agent_.step(action, dt)
        self.__update_data_members()

    def should_stop_wrapped(self, dt, agents, ttc_dp, 
        ego_baseline=None, include_agents_within_range=None, 
        ttc_break_tie=None, return_ttc_tracker=False) -> bool:

        # look up ttc1 (if I stop), ttc2 (if I go)
        # if any of the 4 ttcs are within range, capture that agent
        # (xs, ys)
        ids_ttc_in_range_raw = np.array([]).astype(int)
        for j in range(4):
            # ids_ttc_in_range = np.union1d(ids_ttc_in_range, np.where(ttc_dp[j, int(self.id_), :] < self.default_ttc_)[0])
            ids_ttc_in_range_raw = np.union1d(ids_ttc_in_range_raw, np.where(ttc_dp[j, int(self.id_), :] <= self.ttc_thres_)[0])
        # if we enforce the same observation range as the model
        if include_agents_within_range is not None and include_agents_within_range > 0:
            indices_to_delete = []
            ids_to_delete = [] # for debug
            for i in range(len(ids_ttc_in_range_raw)):
                agent_id = str(ids_ttc_in_range_raw[i])
                poly_dist = calculate_poly_distance(agents['0'], agents[agent_id])
                if (poly_dist > include_agents_within_range):
                    indices_to_delete.append(i)
                    ids_to_delete.append(agent_id)
            ids_ttc_in_range_raw = np.delete(ids_ttc_in_range_raw, indices_to_delete)
            conditional_log(self.log_name_, logger, f'ttc_in_range ids not in distance range: {str(ids_to_delete)} | ids_ttc_in_range_raw={str(ids_ttc_in_range_raw)}', 'debug')
        # for crossing vehicles, only consider my interaction with the head of a series of vehicles in ids_ttc_in_range
        ids_ttc_in_range = []
        for raw_id in ids_ttc_in_range_raw:
            to_add = True
            for selected_id in ids_ttc_in_range:
                selected_agent_lane_id = agents[str(selected_id)].path_[agents[str(selected_id)].closest_lane_id_order_]
                selected_agent_waypoint_idx = agents[str(selected_id)].closest_waypoint_idx_
                raw_agent_lane_id = agents[str(raw_id)].path_[agents[str(raw_id)].closest_lane_id_order_]
                raw_agent_waypoint_idx = agents[str(raw_id)].closest_waypoint_idx_
                # if raw agent is behind selected agent, don't add
                if (selected_agent_lane_id == raw_agent_lane_id and raw_agent_waypoint_idx < selected_agent_waypoint_idx) or \
                    selected_agent_lane_id in self.lane_segments_[raw_agent_lane_id].successors:
                    to_add = False
                    break
                # if selected agent is behind raw agent, remove selected agent
                elif (selected_agent_lane_id == raw_agent_lane_id and raw_agent_waypoint_idx > selected_agent_waypoint_idx) or \
                    raw_agent_lane_id in self.lane_segments_[selected_agent_lane_id].successors:
                    ids_ttc_in_range.remove(selected_id)
            if to_add:
                ids_ttc_in_range.append(raw_id)

        ids_should_stop_for = []

        ttc_trackers = {'min_decision_ttc' : self.default_ttc_, # minimum decision-triggering ttc (if ttc1 > ttc2, should save ttc2)
                        'min_ttc' : self.default_ttc_, # among all agents in ttc range, save their min ttc in (0,1,2,3)
                        'min_ttc0' : self.default_ttc_, # among all agents in ttc range, save their min ttc0
                        'min_ttc1' : self.default_ttc_, # among all agents in ttc range, save their min ttc1
                        'min_ttc2' : self.default_ttc_, # among all agents in ttc range, save their min ttc2
                        'min_ttc3' : self.default_ttc_ # among all agents in ttc range, save their min ttc3
                        }

        should_stop = False
        ego_lane_id = self.path_[self.closest_lane_id_order_]
        ego_waypoint_idx = self.closest_waypoint_idx_
        my_two_degree_successors = get_two_degree_successors(self.lane_segments_, ego_lane_id)
        for other_agent_id, other_agent in agents.items():
            if (other_agent_id == self.id_): 
                continue
            # if other agent is in front of me
            other_agent_lane_id = other_agent.path_[other_agent.closest_lane_id_order_]
            other_agent_waypoint_idx = other_agent.closest_waypoint_idx_
            dist = calculate_poly_distance(self, other_agent)
            # conditional_log(self.log_name_, logger,
            #             f'agent {self.id_} dist to agent {other_agent_id} | dist={dist}', 'info') # DEBUG
            # == front-vehicle logic ==
            if (ego_baseline not in [1,2,5]) and \
                ((other_agent_lane_id == ego_lane_id and ego_waypoint_idx < other_agent_waypoint_idx) or \
                other_agent_lane_id in my_two_degree_successors):
                if (dist <= 1 + self.fad_coeff_ * self.ttc_thres_ * self.v_desired_): # assume ttc_thres_ self stop time if the other agent suddenly stops
                    conditional_log(self.log_name_, logger,
                        f'ego should stop for front agent {other_agent_id} | dist={dist}', 'debug') # DEBUG
                    est_decision_ttc = dist / self.v_desired_
                    ids_should_stop_for.append((int(other_agent_id), est_decision_ttc))
                    should_stop = (should_stop or True) # we want to go through everyone for debug
            else:
                i = int(other_agent_id)
                if i in ids_ttc_in_range:
                    # == back-vehicle logic ==
                    # if this ttc3 agent is behind me, don't consider whether we should stop for them
                    if  (ego_baseline not in [5]) and \
                        ((ego_lane_id == other_agent_lane_id and other_agent_waypoint_idx < ego_waypoint_idx) or \
                         (ego_lane_id in self.lane_segments_[other_agent_lane_id].successors)):
                        conditional_log(self.log_name_, logger, f'ego with agent {i} behind. shouldn\'t stop.', 'debug')
                        should_stop = (should_stop or False)
                    # == ttc logic ==
                    elif (ego_baseline not in [6]):
                        ttc3 = ttc_dp[3, int(self.id_), i] # if we both go
                        ttc1 = ttc_dp[1, int(self.id_), i] # if I stop
                        ttc2 = ttc_dp[2, int(self.id_), i] # if I go
                        ttc0 = ttc_dp[0, int(self.id_), i]

                        # update ttc_trackers
                        min_ttc = min([ttc0, ttc1, ttc2, ttc3])
                        ttc_trackers['min_ttc3'] = min(ttc3, ttc_trackers['min_ttc3'])
                        ttc_trackers['min_ttc2'] = min(ttc2, ttc_trackers['min_ttc2'])
                        ttc_trackers['min_ttc1'] = min(ttc1, ttc_trackers['min_ttc1'])
                        ttc_trackers['min_ttc0'] = min(ttc0, ttc_trackers['min_ttc0'])
                        ttc_trackers['min_ttc'] = min(min_ttc, ttc_trackers['min_ttc'])

                        # max_ttc_go = max(ttc2, ttc3)
                        # criteria = (ttc1 >= max_ttc_go)
                        if ttc1 == ttc2:
                            ttc_trackers['min_decision_ttc'] = min(ttc1, ttc_trackers['min_decision_ttc'])
                            if ego_baseline in [None, 5, 4]:
                                # based on who's agg_level = 0
                                if (ego_baseline not in [4]) and (ttc_break_tie is None or ttc_break_tie == 'agg_level=0'):
                                    # if the other is not mild, I should stop
                                    if other_agent.agg_level_ != 0:
                                        criteria = True 
                                    # if the other is mild, I should go
                                    else:
                                        criteria = False
                                    conditional_log(self.log_name_, logger, 
                                        f'agent {self.id_} ttc_break_tie: {ttc_break_tie} | criteria={criteria}', 'debug')
                                # based on b1
                                elif (ego_baseline not in [4]) and (ttc_break_tie == 'b1'):
                                    conditional_log(self.log_name_, logger, f'agent {self.id_} ttc_break_tie: {ttc_break_tie}', 'debug')
                                    if self.b1_ != other_agent.b1_:
                                        criteria = (self.b1_ < other_agent.b1_)
                                    else:
                                        temp = np.random.random()
                                        if temp < 0.5:
                                            criteria = True
                                        else:
                                            criteria = False
                                    conditional_log(self.log_name_, logger, 
                                        f'agent {self.id_} ttc_break_tie: {ttc_break_tie} | criteria={criteria}', 'debug')
                                # based on coin flip
                                elif ttc_break_tie == 'random':
                                    ttc_sample = np.random.random() # [0,1)
                                    criteria = (ttc_sample < self.stop_prob_)
                                    conditional_log(self.log_name_, logger,
                                        f'agent {self.id_} other_agent {other_agent.id_} ttc_break_tie: {ttc_break_tie} | sample={ttc_sample} | stop_prob = {self.stop_prob_} | criteria={criteria}', 'debug') 
                                elif ttc_break_tie == 'id':
                                    criteria = (int(self.pseudo_id_) < int(other_agent.id_))
                                    log_str = f'agent {self.id_} other_agent {other_agent.id_} agent_baseline={ego_baseline} | ttc_break_tie: {ttc_break_tie} | criteria={criteria} | self.pseudo_id: {self.pseudo_id_}'
                                    conditional_log(self.log_name_, logger, log_str, 'debug') 
                                else:
                                    raise Exception(f"invalid ttc_break_tie: {ttc_break_tie}")
                            else:
                                if ego_baseline in [0,1]:
                                    temp = np.random.random()
                                    if temp < 0.5:
                                        criteria = True
                                    else:
                                        criteria = False
                                else:
                                    raise Exception(f"invalid ego_baseline: {ego_baseline}")
                        else:
                            criteria = (ttc1 > ttc2)
                            ttc_trackers['min_decision_ttc'] = min(ttc2, ttc_trackers['min_decision_ttc'])
                        # debug
                        if criteria:
                            ids_should_stop_for.append((i, ttc2))
                        
                        should_stop = (should_stop or criteria) # we could break loop here, but calculate all so we can debug
                        conditional_log(self.log_name_, logger,
                                    f'agent {self.id_} (agg={self.agg_level_}) ttc with agent {i} (agg={other_agent.agg_level_}) | ttc3 (both go) = {ttc3} | ttc1 (I stop) = {ttc1} | ttc2 (I go) = {ttc2}', 'debug') 
                        if ego_baseline is None:
                            conditional_log(self.log_name_, logger,
                                    f'agent {self.id_} (agg={self.agg_level_}) with agent {i} | ttc logic | should stop: {criteria}', 'debug') 
                        else:
                            conditional_log(self.log_name_, logger,
                                    f'agent {self.id_} (agg={self.agg_level_}) with agent {i} | ttc logic | ego_baseline {ego_baseline} should_stop: {criteria}', 'debug') 
                    
        if should_stop:
            conditional_log(self.log_name_, logger,
                        f'agent {self.id_} should stop for: {str(ids_should_stop_for)}', 'debug') 
    
        self.__update_data_members()

        if return_ttc_tracker:
            return should_stop, ids_should_stop_for, ttc_trackers
        else:
            return should_stop, ids_should_stop_for

    def __update_data_members(self):
        self.__dict__.update(self.saved_agent_.__dict__)
