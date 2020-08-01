# Uses Argoverse v1.1 : https://github.com/argoai/argoverse-api

import math, sys
import numpy as np
import pickle
from argoverse.utils.se3 import SE3
from datetime import datetime
from gym_road_interactions.core import AgentType, Position, Agent, ObservableState, Observation

def wrap_to_pi(input_radian: float) -> float:
    """
    Helper function to wrap input radian value to [-pi, pi) (+pi exclusive!)
    """
    return ((input_radian + math.pi) % (2 * math.pi) - math.pi)

def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Helper function to generate a rotation matrix that rotates w.r.t z-axis
    Args:
        angle (float): angle in radian
    """
    return np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0],
                     [0, 0, 1]])

def rotation_matrix_x(angle: float) -> np.ndarray:
    """
    Helper function to generate a rotation matrix that rotates w.r.t x-axis
    Args:
        angle (float): angle in radian
    """
    return np.array([[1, 0, 0],
                     [0, math.cos(angle), math.sin(angle)],
                     [0, -math.sin(angle), math.cos(angle)]])

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def create_bbox_world_frame(agent : 'Agent', pos : 'Position'=None) -> np.ndarray:
    """
    pos (optional): if provided, use this position instead of agent position
    """
    if pos is None:
        x = agent.observable_state_.position_.x_
        y = agent.observable_state_.position_.y_
        heading = agent.observable_state_.position_.heading_
    else:
        x = pos.x_
        y = pos.y_
        heading = pos.heading_
    bbox_object_frame = np.array(
        [
            [agent.length_ / 2.0, agent.width_ / 2.0, agent.height_ / 2.0],
            [agent.length_ / 2.0, -agent.width_ / 2.0, agent.height_ / 2.0],
            [-agent.length_ / 2.0, agent.width_ / 2.0, agent.height_ / 2.0],
            [-agent.length_ / 2.0, -agent.width_ / 2.0, agent.height_ / 2.0],
        ]
    )
    agent_to_city_frame_se3 = SE3(rotation=rotation_matrix_z(heading), 
                                    translation=np.array([x, y, 0]))
    bbox_in_city_frame = agent_to_city_frame_se3.transform_point_cloud(bbox_object_frame)
    return bbox_in_city_frame

# Logging function
def log(fname, s):
    # if not os.path.isdir(os.path.dirname(fname)):
    #     os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    f.write(f'{str(datetime.now())}: {s}\n')
    f.close()

# helper to choose whether to log to a file or logger out
# logger_level: [info, debug, warning, error]
def conditional_log(log_name, logger, content, logger_level='debug'):
    if log_name is not None:
        log(log_name, content)
    else:
        if logger_level == 'debug':
            logger.debug(content)
        elif logger_level == 'info':
            logger.info(content)
        else:
            logger.warning(f'conditional_log called with invalid logger_level={logger_level}, showing content with info')
            logger.info(content)

def remap(v, x, y):
    # v: value / np array
    # x: original range
    # y: target range
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])

