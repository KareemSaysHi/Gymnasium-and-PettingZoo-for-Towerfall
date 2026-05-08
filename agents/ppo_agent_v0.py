import logging
import random
from typing import Any, Mapping
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import torch
from towerfall import Connection

class PPOAgent_V0():
  '''
  A minimal agent that works with the PettingZoo Env.

  params connection: A connection to a Towerfall game.
  params attack_archers: If True, the agent will attack other neutral archers.
  '''
  def __init__(self, connection: Connection, attack_archers: bool = False, model = None):
    self.state_init: Mapping[str, Any] = {}
    self.state_scenario: Mapping[str, Any] = {}
    self.state_update: Mapping[str, Any] = {}
    self.pressed = set()
    self.connection = connection
    self.attack_archers = attack_archers

    #set the model's parameters from the tuned model
    self.model = model
    self.id = 0
    self.frames_per_action = 5

    self.my_pos = None
    self.my_vel = None
    self.your_pos = None
    self.your_vel = None
    self.my_arrows = None
    self.your_arrows = None

    self.direction = None
    self.button = None

    self.dir_to_input = ['', 'l', 'r', 'u', 'd']
    self.but_to_input = ['', 'j', 'z']





  def act(self, game_state: Mapping[str, Any]):
    '''
    Handles a game message.
    '''

    # There are three main types to handle, 'init', 'scenario' and 'update'.
    # Check 'type' to handle each accordingly.
    if game_state['type'] == 'init':
      # 'init' is sent every time a match series starts. It contains information about the players and teams.
      # The seed is based on the bot index so each bots acts differently.
      self.state_init = game_state
      random.seed(self.state_init['index'])
      # Acknowledge the init message.
      self.connection.send_json(dict(type='result', success=True))
      return True

    if game_state['type'] == 'scenario':
      # 'scenario' informs your bot about the current state of the ground. Store this information
      # to use in all subsequent loops. (This example bot doesn't use the shape of the scenario)
      self.state_scenario = game_state
      # Acknowledge the scenario message.
      self.connection.send_json(dict(type='result', success=True))
      return

    if game_state['type'] == 'update':
      # 'update' informs the state of entities in the map (players, arrows, enemies, etc).
      self.update_data(game_state)

      #IF IT IS TIME TO ACT:
      #   take all observations and make it into the proper size:
      #   then put it into the model, output a tensor, that's our new action vector

      if self.id % self.frames_per_action == 0:

        observations = []
        observations.append(self.my_pos['x'])
        observations.append(self.my_pos['y'])
        observations.append(self.my_vel['x'])
        observations.append(self.my_vel['y'])
        observations.append(self.your_pos['x'])
        observations.append(self.your_pos['y'])
        observations.append(self.your_vel['x'])
        observations.append(self.your_vel['y'])
        observations = np.array(observations)

        action, _ = self.model.predict(observations, deterministic=False)
        print(action)
        self.direction = action[0]
        self.button = action[1]

        obs_tensor = torch.as_tensor(observations).unsqueeze(0).float().to(self.model.policy.device)
        distribution = self.model.policy.get_distribution(obs_tensor)
        for i, dist in enumerate(distribution.distribution):
            print(f"dim {i} probs: {dist.probs.detach().cpu().numpy()}")

      if self.id % self.frames_per_action == self.frames_per_action-1:
        self.pressed.clear()
        self.send_actions()
      else:
      #then act on it
        if self.direction > 0:
          self.press(self.dir_to_input[self.direction])
        if self.button > 0:
          self.press(self.but_to_input[self.button])
        self.send_actions()

  def update_data(self, game_state: Mapping[str, Any]):
    '''
    Refresh cached archer data (pos, vel, dead, arrows) from an 'update' message.
    Safe to call with any message — non-'update' types are ignored so cached
    values stay intact. self.dead starts False and only flips to True when we
    either see dead=True on the archer or the archer is missing from entities.
    '''
    if game_state.get('type') != 'update':
      return
    self.state_update = game_state
    self.id = game_state['id']
    #self.my_pos = None
    #self.my_vel = None
    #self.your_pos = None
    #self.your_vel = None
    #self.my_arrows = 0
    #self.your_arrows = 0

    #print(self.state_update)

    for e in game_state['entities']:
      if e['type'] == 'archer' and e['playerIndex'] == self.state_init['index']:
        self.my_pos = e['pos']
        self.my_vel = e['vel']
        self.my_arrows = len(e.get('arrows', []))
        
      if e['type'] == 'archer' and e['playerIndex'] != self.state_init['index']:
        self.your_pos = e['pos']
        self.your_vel = e['vel']
        self.your_arrows = len(e.get('arrows', []))
        
    return

  def press(self, b):
    self.pressed.add(b)

  def send_actions(self):
    assert self.state_update
    self.connection.send_json(dict(
      type = 'actions',
      actions = ''.join(self.pressed),
      id = self.state_update['id']
    ))
    self.pressed.clear()
