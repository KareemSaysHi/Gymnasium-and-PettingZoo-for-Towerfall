import random
from typing import Any, Mapping
import numpy as np
from stable_baselines3.ppo import MlpPolicy
import torch
from towerfall import Connection


import env_methods as env_methods

'''
An agent that acts based on a model's outputs.  A normalizer pickle
file is required as well.
'''
class ModelAgent():
 
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