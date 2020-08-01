from enum import Enum
import math
import copy
import numpy as np
from typing import List, Optional, Sequence

# b
class AgentType(Enum):
    ego_vehicle = 0
    other_vehicle = 1
    pedestrian = 2
    cyclist = 3
    motorcycle = 4
    on_road_obstacle = 5
    other_mover = 6

# c
class MapCategory(Enum):
    driveway = 0
    intersection = 1
    crosswalk = 2
    sidewalk = 3

class Position:
    def __init__(self, x: float, y: float, heading: float):
        """
        Represents a position vector p = [x, y, heading]
        Args:
            x, y: city coordinates, in meters
            heading: heading angle w.r.t. city coordinates, in radians
        """
        self.x_ = x
        self.y_ = y
        self.heading_ = ((heading + math.pi) % (2 * math.pi) - math.pi)
        self.world_to_position_se3_ = None
    
    def calculate_distance(self, other_position: 'Position') -> float:
        """
        Calculates the distance between self and other position
        """
        dist = math.sqrt((other_position.x_ - self.x_)**2 + \
                         (other_position.y_ - self.y_)**2)
        return dist

class ObservableState:
    def __init__(self, position: Position, velocity: float, yaw_rate: float, turn_signal: int = 0, stop_light: int = 0):
        """
        Represents an observable state vector s = [p, v]
        Args:
            position: position [x, y, heading]
            velocity: longitudinal velocity, in m/s
            yaw_rate: change in yaw, in rad/s
            turn_signal: 0 none, -1 left, 1 right
            stop_light: 0 off, 1 on
        """
        self.position_ = position
        self.velocity_ = velocity
        self.yaw_rate_ = yaw_rate
        self.turn_signal_ = turn_signal
        self.stop_light_ = stop_light

class Observation:
    def __init__(self, observable_state: ObservableState, 
                 agent_type: AgentType, lane_type: str = None):
        """
        Represents an observation vector o = [s, b]
        Args:
            observable_state: the observable state of the corresponding agent
            agent_type: the type of the corresponding agent
        """
        self.observable_state_ = observable_state
        self.agent_type_ = agent_type
        # This optional string contains info about the lane segment property 
        # (eg. roundabout, straight_lane, transition)
        self.lane_type_ = lane_type

class MapRange:
    def __init__(self, x_min: float, x_max: float, 
                 y_min: float, y_max: float, city_name: str):
        """
        Represents the range of map of concern in a world
        Args:
            x_min, x_max, y_min, y_max: city coordinates of the map boundary
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        """
        self.x_min_ = x_min
        self.x_max_ = x_max
        self.y_min_ = y_min
        self.y_max_ = y_max
        self.city_name_ = city_name

class LaneSegment:
    def __init__(
        self,
        id: int, # in neighborhood-v0 it's str
        centerline: np.ndarray,
        length: float,
        priority: Optional[int] = None,
        priority2: Optional[int] = None,
        curve_center: Optional[np.ndarray] = None,
        curve_radius: Optional[float] = None,
        turn_direction: Optional[str] = None,
        lane_heading: Optional[float] = None,
        is_intersection: Optional[bool] = False,
        intersection_id: Optional[str] = None,
        has_traffic_control: Optional[bool] = None,
        l_neighbor_id: Optional[int] = None,
        r_neighbor_id: Optional[int] = None,
        predecessors: Optional[Sequence[int]] = [],
        successors: Optional[Sequence[int]] = [],
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way"
            centerline: The coordinates of the lane segment's center line.
            length: length of this lane
            priority: integer indicating lane priority. Larger is more important
            curve_center: circle center of the circle which this curve is a part of
            curve_radius: radius of the circle which this curve is a part of
            has_traffic_control: T/F
            turn_direction: 'right', 'left', or None
            lane_heading: heading of the direction of a straight lane in radian (eg. S->N lane is pi/2)
            is_intersection: Whether or not this lane segment is an intersection - if yes, the intersection_id is filled in
            intersection_id: intersection_id
            l_neighbor_id: Unique ID for left neighbor
            r_neighbor_id: Unique ID for right neighbor
            predecessors: The IDs of the lane segments that come after this one
            successors: The IDs of the lane segments that come before this one.
           
        """
        self.id = id
        self.centerline = centerline
        self.length = length
        self.priority = priority
        self.priority2 = priority2
        self.curve_center = curve_center
        self.curve_radius = curve_radius
        self.has_traffic_control = has_traffic_control
        self.turn_direction = turn_direction
        self.lane_heading = lane_heading
        self.is_intersection = is_intersection
        self.intersection_id = intersection_id
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.predecessors = predecessors
        self.successors = successors

class Agent:
    def __init__(self, 
                 id: str,
                 observable_state: ObservableState, 
                 agent_type: AgentType, 
                 goal: ObservableState, 
                 width: float,
                 length: float,
                 height: float,
                 observations: list = None,
                 map_category: MapCategory = None,
                 use_saved_path: bool = False,
                 path: list = None,
                 closest_lane_id_order: int = -1,
                 closest_waypoint_idx: int = -1):
        """
        Represents an agent
        Args:
            id: unique id representing this agent
            observable_state: observable state of the agent itself
            agent_type: agent type
            goal: observable state of the goal
            width, length: lateral and longitudinal size in meters
            observations: list of Observations, observations within a certain range. Default empty list
            map_category: map category of the current location of the agent. Default None.
            use_saved_trajectory: whether to update this agent using saved trajectory.
            path: list of np.ndarray, each represents a lane centerline
            closest_lane_id_order: the order of the currently closest lane_id in path. Should be updated if path is set.
        """
        self.id_ = id
        self.observable_state_ = observable_state
        self.agent_type_ = agent_type
        self.goal_ = goal
        self.closest_lane_id_order_ = closest_lane_id_order
        self.closest_waypoint_idx_ = closest_waypoint_idx
        if observations is not None:
            self.observations_ = observations
        else:
            self.observations_ = []
        self.map_category_ = map_category
        # radius in which the agent observes others
        self.observation_radius_ = 20.0
        # size
        self.width_ = width # lateral
        self.length_ = length # longitudinal
        self.height_ = height # vertical
        # use saved trajectory?
        self.use_saved_path_ = use_saved_path
        if path is not None:
            self.path_ = path
        else:
            self.path_ = []
        # bounding box in world frame
        self.bbox_world_frame_ = None

    def set_path(self, path: List[str]) -> None:
        """
        Save path
        Args:
            path: List[str]
        """
        if not (self.use_saved_path_):
            raise Exception('Cannot set path of an agent that doesn\'t use saved path')
        self.path_ = path
    
    def set_goal(self, goal: ObservableState) -> None:
        self.goal_ = goal

