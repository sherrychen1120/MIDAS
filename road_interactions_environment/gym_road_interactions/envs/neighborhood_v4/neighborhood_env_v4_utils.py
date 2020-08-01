# Closed neighborhood environment with a roundabout, 4 t-intersections
# all global util functions

# Python
import pdb, copy, os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import logging
from queue import Queue
from typing import Any, Dict, Tuple, List
import cv2, time
# Argoverse
from argoverse.utils.se3 import SE3
# gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding
# utils
from gym_road_interactions.utils import create_bbox_world_frame, wrap_to_pi
from gym_road_interactions.viz_utils import visualize_agent
from gym_road_interactions.core import AgentType, Position, Agent, ObservableState, Observation, LaneSegment
from shapely.geometry import Point, Polygon

def calculate_remaining_lane_distance(lane_id: int, curr_pos: Position, lane_segments: List[LaneSegment]) -> float:
    # calculates the distance between curr_pos and lane end
    curr_xy = np.array([curr_pos.x_, curr_pos.y_])
    lane_end_xy = lane_segments[lane_id].centerline[-1,:]
    l = np.linalg.norm(lane_end_xy - curr_xy) # straight-line distance
    
    # straight lanes
    if lane_segments[lane_id].curve_center is None:
        remain_lane_distance = l
    else:
        r = lane_segments[lane_id].curve_radius
        remain_theta = math.acos((2*r*r - l**2)/(2*r*r))
        remain_lane_distance = remain_theta * r

    return remain_lane_distance

def calculate_traversed_lane_distance(lane_id: int, curr_pos: Position, lane_segments: List[LaneSegment]) -> float:
    # calculates the distance between curr_pos and lane start
    curr_xy = np.array([curr_pos.x_, curr_pos.y_])
    lane_start_xy = lane_segments[lane_id].centerline[0,:]
    l = np.linalg.norm(lane_start_xy - curr_xy) # straight-line distance
    
    # straight lanes
    if lane_segments[lane_id].curve_center is None:
        traversed_lane_distance = l
    else:
        r = lane_segments[lane_id].curve_radius
        traversed_theta = math.acos((2*r*r - l**2)/(2*r*r))
        traversed_lane_distance = traversed_theta * r

    return traversed_lane_distance

def detect_future_collision(agent1 : Agent, agent2: Agent, future_pos1 : Position, future_pos2: Position) -> bool:
    dist = math.sqrt((future_pos1.x_ - future_pos2.x_)**2 + \
                     (future_pos1.y_ - future_pos2.y_)**2)
    return (dist < agent1.radius_ + agent2.radius_ + 1.0)

def detect_collision(agent1: Agent, agent2: Agent) -> bool:
    dist = math.sqrt((agent1.observable_state_.position_.x_ - agent2.observable_state_.position_.x_)**2 + \
                     (agent1.observable_state_.position_.y_ - agent2.observable_state_.position_.y_)**2)
    return (dist < agent1.radius_ + agent2.radius_ + 1.0)

def calculate_poly_distance(agent1: Agent, agent2: Agent) -> float:
    dist = math.sqrt((agent1.observable_state_.position_.x_ - agent2.observable_state_.position_.x_)**2 + \
                     (agent1.observable_state_.position_.y_ - agent2.observable_state_.position_.y_)**2)
    return dist - agent1.radius_ - agent2.radius_

def calculate_time_to_collision(agent1: Agent, agent2: Agent, default_ttc: float, 
                                dt: float, assume_vel1: int = 0, assume_vel2: int = 0):
    obs_window = 10 # number of timesteps to look forward # always set this to be the same as largest ttc thres
    time_to_collision = default_ttc

    for ts in range(1, obs_window+1):
        time_period = dt * ts
        pred_pos1,_,_ = agent1.drive_along_path(time_period, assume_vel=assume_vel1)
        pred_pos2,_,_ = agent2.drive_along_path(time_period, assume_vel=assume_vel2)
        if detect_future_collision(agent1, agent2, pred_pos1, pred_pos2):
            time_to_collision = time_period
            break
    
    # DEBUG
    # print(f'raw ttc: from agent{agent1.id_} to agent{agent2.id_}: {time_to_collision}')
    return time_to_collision

def get_two_degree_successors(lane_segments, lane_id):
    first_degree_successors = np.array(lane_segments[lane_id].successors)
    second_degree_successors = np.array([])
    for suc in first_degree_successors:
        second_degree_successors = np.union1d(second_degree_successors, np.array(lane_segments[suc].successors))
    all_successors = np.union1d(first_degree_successors, second_degree_successors)
    return all_successors