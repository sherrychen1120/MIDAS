# MIDAS: Multi-agent Interaction-aware Decision-making with Adaptive Strategies for Urban Autonomous Navigation
Xiaoyi Chen, Pratik Chaudhari

GRASP Lab, University of Pennsylvania

ArXiV: https://arxiv.org/abs/2008.07081

## Prepare the environment
Install Argoverse following the instructions here: https://github.com/argoai/argoverse-api

Install `ffmpeg`

## Create collision and interaction sets
1. To create collision sets, change `na` on line 6 to be the number of agents in the environment, and change `date` on line 7 of `road_interactions_environment/neighhood_v4_collision_set_gen.py`. Then run `python road_interactions_environment/neighhood_v4_collision_set_gen.py`.
2. To create interaction sets, follow the steps in `road_interactions_environment/neighhood_v4_interaction_set_creation.ipynb`.

## Train the model
In `policy_network/neighborhood_v4_ddqn/train_tr5.py`:
1. Change the filepaths on lines 237-243 to point to your generated collision sets, interaction set and evaluation set.
2. Change the environment and training hyperparameters from line 48 to 158 for your training purposes. The default values are for MIDAS. In order to run MLP, DeepSet, SocialAttention with the same hyperparameters, simply change the value of `value_net` on line 119 to `vanilla`, `deep_set` or `social_attention`.
3. Run `python policy_network/neighborhood_v4_ddqn/train_tr5.py`. Arguments:
```
    --date Training date
    --code ID of your experiment. Eg. c0-0
    --seed Experiment seed. Any integer between 0 and 65535.
```

## Visualize an episode with a model checkpoint
In `policy_network/neighborhood_v4_ddqn/visualize_episode.py`:
1. Update the variables on lines 199-218 depending on the date, checkpoint ID and filepath, dataset filepath and the ids of the episodes that you want to visualize.
2. Run `python policy_network/neighborhood_v4_ddqn/visualize_episode.py`

# Code References
Argoverse https://github.com/argoai/argoverse-api

Set Transformer https://github.com/juho-lee/set_transformer

# License
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
