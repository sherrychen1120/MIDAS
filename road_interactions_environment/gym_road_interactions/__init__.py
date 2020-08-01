from gym.envs.registration import register

register(
    id='Neighborhood-v4',
    entry_point='gym_road_interactions.envs.neighborhood_v4:NeighborhoodEnvV4',
)
