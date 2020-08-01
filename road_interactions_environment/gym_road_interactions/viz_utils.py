# Adapted from Argoverse v1.1 : https://github.com/argoai/argoverse-api
# argoverse-api/demo_usage/visualize_30hz_benchmark_data_on_map.py

import math
import logging
import pdb
from typing import Tuple

import imageio
# all mayavi imports MUST come before matplotlib, else Tkinter exceptions
# will be thrown, e.g. "unrecognized selector sent to instance"
import mayavi
import matplotlib.pyplot as plt
import numpy as np

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.geometry import filter_point_cloud_to_polygon, rotate_polygon_about_pt
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons, plot_bbox_2D
from argoverse.utils.se3 import SE3
from argoverse.utils.subprocess_utils import run_command

from gym_road_interactions.core import Agent, AgentType
from gym_road_interactions.utils import rotation_matrix_z

logger = logging.getLogger(__name__)

def render_map(
        city_name: str,
        ax: plt.Axes,
        axis: str,
        local_lane_polygons: np.ndarray,
        local_das: np.ndarray,
        local_lane_centerlines: list, # list of np.ndarray
        city_to_egovehicle_se3: SE3,
        avm: ArgoverseMap) -> None:
    """
    Helper function to draw map
    """
    if axis is not "city_axis":
        # rendering instead in the egovehicle reference frame
        for da_idx, local_da in enumerate(local_das):
            local_da = city_to_egovehicle_se3.inverse_transform_point_cloud(local_da)
            local_das[da_idx] = rotate_polygon_about_pt(local_da, city_to_egovehicle_se3.rotation, np.zeros(3))

        for lane_idx, local_lane_polygon in enumerate(local_lane_polygons):
            local_lane_polygon = city_to_egovehicle_se3.inverse_transform_point_cloud(local_lane_polygon)
            local_lane_polygons[lane_idx] = rotate_polygon_about_pt(
                local_lane_polygon, city_to_egovehicle_se3.rotation, np.zeros(3)
            )

    draw_lane_polygons(ax, local_lane_polygons)
    draw_lane_polygons(ax, local_das, color="tab:pink")

    for lane_cl in local_lane_centerlines:
        ax.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="grey", alpha=1, linewidth=1, zorder=0)

def visualize_agent(
        ax: plt.Axes,
        agent: Agent,
        color: Tuple[float, float, float] = None
        ) -> None:
    """
    Helper function to draw bounding box and arrow of an agent
    """
    # draw center dot
    # TODO: change color based on agent type
    x = agent.observable_state_.position_.x_
    y = agent.observable_state_.position_.y_
    # change color based on agent type
    if (agent.agent_type_ == AgentType.ego_vehicle):
        logger.debug(f'Rendering ego vehicle at {x}, {y}')
    clr_lib = {AgentType.ego_vehicle : (0,1,1), # cyan
               AgentType.other_vehicle : (0,0,1), # blue
               AgentType.pedestrian : (1,0,0), # red
               AgentType.cyclist : (0,1,0), # green
               AgentType.motorcycle : (0.196, 0.8, 0.196), # limegreen
               AgentType.on_road_obstacle : (0,0,0), # black
               AgentType.other_mover : (1,0,1)} # magenta
    ax.scatter(x, y, 100, color=clr_lib[agent.agent_type_], marker=".", zorder=2)
    # draw arrow
    heading = agent.observable_state_.position_.heading_
    scale_factor = math.log(abs(agent.observable_state_.velocity_) + 1.0)
    dx = scale_factor * math.cos(heading)
    dy = scale_factor * math.sin(heading)
    ax.arrow(x, y, dx, dy, color="r", width=0.2, zorder=2)
    if agent.id_ == '0':
        ax.annotate(agent.id_, (x, y), color='k', weight='bold', fontsize=7, ha='center', va='center')
    else:
        ax.annotate(agent.id_, (x, y), color='w', weight='bold', fontsize=7, ha='center', va='center')
    # pdb.set_trace()
    # draw bounding box
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
    if color is not None:
        plot_bbox_2D(ax, bbox_in_city_frame, color)
    else:
        plot_bbox_2D(ax, bbox_in_city_frame, clr_lib[agent.agent_type_])

def write_nonsequential_idx_video(img_wildcard: str, output_fpath: str, fps: int, return_cmd: bool = False) -> None:
    """
    Args:
        img_wildcard: string
        output_fpath: string
        fps: integer, frames per second

    Returns:
       None
    """
    cmd = f"ffmpeg -r {fps} -f image2 -pattern_type glob -i \'{img_wildcard}\'"
    cmd += " -vcodec libx264 -profile:v main"
    cmd += " -level 3.1 -preset medium -crf 23 -x264-params ref=4 -acodec"
    cmd += f" copy -movflags +faststart -pix_fmt yuv420p  -vf scale=920:-2"
    cmd += f" {output_fpath}"
    print(cmd)
    stdout_data, stderr_data = run_command(cmd, True)
    print(stdout_data)
    print(stderr_data)

    if return_cmd:
        return cmd
