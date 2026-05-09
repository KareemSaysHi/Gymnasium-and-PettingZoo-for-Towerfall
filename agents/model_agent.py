import random
from typing import Any, Mapping
import numpy as np
from stable_baselines3.ppo import MlpPolicy
import torch
from towerfall import Connection


import env_methods as env_methods

class ModelAgent():
  '''
  A minimal agent that works with the PettingZoo Env.

  params connection: A connection to a Towerfall game.
  params attack_archers: If True, the agent will attack other neutral archers.

  mode can either be training or evaluating
  '''
  def __init__(self, model = None, normalizer = None):
    
    self.direction_pressed = ''
    self.button_pressed = ''

    # model information
    self.model = model
    self.normalizer = normalizer
    self.normalizer.training = False
    self.normalizer.norm_reward = False

  #given an observation, make an action
  def take_action(self, observation):

    normalized_obs = self.normalizer.normalize_obs(observation)
    action, _ = self.model.predict(normalized_obs, deterministic=False)

    self.direction = action[0]
    self.button = action[1]

    return np.array([self.direction, self.button])

    #some debugging things:
    #obs_tensor = torch.as_tensor(observations).unsqueeze(0).float().to(self.model.policy.device)
    #distribution = self.model.policy.get_distribution(obs_tensor)
    #for i, dist in enumerate(distribution.distribution):
    #    print(f"dim {i} probs: {dist.probs.detach().cpu().numpy()}")