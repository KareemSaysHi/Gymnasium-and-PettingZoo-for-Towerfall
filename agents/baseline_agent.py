import random
import numpy as np


'''
BaselineAgent is a simple agent used for models to train against.
BaselineAgent is inspired by TowerfallAI's SimpleAgent
'''
class BaselineAgent:
  def __init__(self):
    self.direction_pressed = ''
    self.button_pressed = ''

    self.dir_to_input = ['', 'u', 'ur', 'r', 'rd', 'd', 'ld', 'l', 'ul'] #r = right, l = left, u = up, d = down
    self.but_to_input = ['', 'j', 'z', 's'] #j = jump, z = dash, s = shoot

  #given an observation, make an action
  def take_action(self, observation):

    self.direction_pressed = ''
    self.button_pressed = ''

    my_pos = {
      'x': observation['my_movement'][0],
      'y': observation['my_movement'][1]
    }

    enemy_pos = {
      'x': observation['your_movement'][0],
      'y': observation['your_movement'][1]
    }

    # Generally move toward opponent
    if my_pos['x'] < enemy_pos['x']:
      self.direction_pressed = 'r' if random.randint(0, 3) > 0 else 'l'
    else:
      self.direction_pressed = 'l' if random.randint(0, 3) > 0 else 'r'


    # If in the same line shoot,
    if abs(my_pos['y'] - enemy_pos['y']) < 20:
      if random.randint(0, 5) == 0:
        self.button_pressed = 's'

    # Presses dash randomly.
    if random.randint(0, 5) == 0:
      self.button_pressed = 'z'

    # Presses jump randomly.
    if random.randint(0, 10) == 0:
      self.button_pressed = 'j'

    return np.array([
      self.dir_to_input.index(self.direction_pressed), 
      self.but_to_input.index(self.button_pressed)
    ])
