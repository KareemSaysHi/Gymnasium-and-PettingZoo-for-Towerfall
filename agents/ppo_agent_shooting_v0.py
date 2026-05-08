import random
from typing import Any, Mapping
import numpy as np
from stable_baselines3.ppo import MlpPolicy
import torch
from towerfall import Connection

import env_methods as env_methods

class PPOAgentShooting_V0():
  '''
  A minimal agent that works with the PettingZoo Env.

  params connection: A connection to a Towerfall game.
  params attack_archers: If True, the agent will attack other neutral archers.

  mode can either be training or evaluating
  '''
  def __init__(self, connection: Connection, mode = "training", model = None, normalizer = None, frames_per_action = 5):
    
    #Necessary TowerFall parameters
    self.state_init = {}
    self.state_scenario = {}
    self.state_update = {}
    self.pressed = set()
    self.connection = connection
    self.attack_archers = True
    self.id = 0

    #mode and model information
    self.mode = mode
    self.model = model
    
    self.normalizer = normalizer

    if self.mode == "evaluating":
      self.normalizer.training = False
      self.normalizer.norm_reward = False

    self.frames_per_action = frames_per_action
    
    #game state variables
    self.position = None
    self.velocity = None
    self.my_movement = None
    self.my_arrows = None
    self.your_movement = None
    self.your_arrows = None
    self.my_facing = None
    self.dead = None

    #variables necessary for action masking
    self.dodge_cooldown = 0
    self.can_dodge = True
    self.can_jump = True
    self.held_jump_on_last_frame = False
    self.on_ground = True
    self.on_wall = False

    self.direction = None
    self.button = None

    self.dir_to_input = ['', 'u', 'ur', 'r', 'rd', 'd', 'ld', 'l', 'ul']
    self.but_to_input = ['', 'j', 'z', 's']

  def reset_vars(self):
    self.can_dodge = True
    self.can_jump = True
    self.held_jump_on_last_frame = False
    self.on_ground = True
    self.on_wall = False
    self.direction = 0
    self.button = 0
    
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

      if self.mode == "training":
        return #if we are training, no action should be done from a model.
      
      #if we are not training, then have our model decide what to do:
      #   take all observations and make it into the proper size:
      #   then put it into the model, output a tensor, that's our new action vector

      if self.id % self.frames_per_action == 0:
        
        observations = env_methods.build_obs(self) #is this all I need?  Or do I need the other player as well...
        normalized_obs = self.normalizer.normalize_obs(observations)
        action, _ = self.model.predict(normalized_obs, deterministic=True)
        print(action)
        self.direction = action[0]
        self.button = action[1]

        #obs_tensor = torch.as_tensor(observations).unsqueeze(0).float().to(self.model.policy.device)
        #distribution = self.model.policy.get_distribution(obs_tensor)
        #for i, dist in enumerate(distribution.distribution):
        #    print(f"dim {i} probs: {dist.probs.detach().cpu().numpy()}")

      return self.direction, self.button

      if self.direction > 0:
        self.press(self.dir_to_input[self.direction])
      if self.button > 0:
        self.press(self.but_to_input[self.button])

      if self.id % self.frames_per_action == self.frames_per_action-1:
        self.remove('s')

      self.send_actions()

  def update_data(self, game_state: Mapping[str, Any]):
    
    if game_state.get('type') != 'update':
      return
    self.state_update = game_state
    self.id = game_state['id']

    self.dead = True #will correct later in function

    for e in game_state['entities']:
      if e['type'] == 'archer' and e['playerIndex'] == self.state_init['index']:
        self.position = [e['pos']['x'], e['pos']['y']]
        self.velocity = [e['vel']['x'], e['vel']['y']]
        self.my_movement = self.position + self.velocity
        self.my_arrows = len(e.get('arrows', []))
        self.my_facing = (e['facing'] + 1) // 2
        #self.dodge_cooldown = e['dodgeCooldown']
        #self.can_dodge = (self.dodge_cooldown == 0)
        self.on_ground = e['onGround']
        self.on_wall = e['onWall']
        self.dead = False
        #we can jump if we are (on the ground or wall), or if we held the button last frame and are moving upward
        self.can_jump = ((self.on_ground or self.on_wall) or ((e['vel']['y'] > 0) and self.held_jump_on_last_frame))
        
      if e['type'] == 'archer' and e['playerIndex'] != self.state_init['index']:
        self.your_movement = [e['pos']['x'], e['pos']['y'], e['vel']['x'], e['vel']['y']]
        self.your_arrows = len(e.get('arrows', []))        
    return

  
  #press a button
  def press(self, b):
    self.pressed.add(b)
    
  #un-press a button  
  def remove(self, b):
    if b in self.pressed:
      self.pressed.remove(b)

  #send raw buttons to main server
  def send_actions(self):
    assert self.state_update
    self.connection.send_json(dict(
      type = 'actions',
      actions = ''.join(self.pressed),
      id = self.state_update['id']
    ))
    self.pressed.clear()

  #turn env.action_space output into buttons and send
  def send_env_actions(self, direction, button, remove_shoot = False):
    if direction > 0:
      for c in self.dir_to_input[direction]:
        self.press(c)
    if button > 0:
      self.press(self.but_to_input[button])

    if 'j' in self.but_to_input[button]:
      self.held_jump_on_last_frame = True #we jumped
    else:
      self.held_jump_on_last_frame = False
     
    if remove_shoot:
      self.remove('s') #make agent fire on final frame of action
    self.send_actions()
