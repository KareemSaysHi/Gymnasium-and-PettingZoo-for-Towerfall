import random
import numpy as np

class SmarterBaselineAgent:
  def __init__(self):
    self.direction_pressed = ''
    self.button_pressed = ''

    self.dir_to_input = ['', 'u', 'ur', 'r', 'rd', 'd', 'ld', 'l', 'ul']
    self.but_to_input = ['', 'j', 'z', 's']

  #given an observation, make an action
  #r = right, l = left, u = up, d = down, j = jump, z = dash, s = shoot
  def take_action(self, observation):
    
    # This bot acts based on the position of the other player only. It
    # has a very random play style:
    #  - Runs to the enemy when they are below.
    #  - Runs away from the enemy when they are above.
    #  - Shoots when in the same horizontal line.
    #  - Dashes randomly.
    #  - Jumps randomly.

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

    dangerous_arrow = {
      'exists': observation['arrow0_exists'],
      'x': observation['arrow0_pos-vel'][0],
      'y': observation['arrow0_pos-vel'][1],
      'vx': observation['arrow0_pos-vel'][2],
      'vy': observation['arrow0_pos-vel'][3] 
    }

    if my_pos['x'] < enemy_pos['x']:
      self.direction_pressed = 'r' if random.randint(0, 3) > 0 else 'l'
    else:
      self.direction_pressed = 'l' if random.randint(0, 3) > 0 else 'r'


    # If in the same line shoots,
    if abs(my_pos['y'] - enemy_pos['y']) < 20:
      if random.randint(0, 5) == 0:
        self.button_pressed = 's'

    # Dashes if arrow is close:
    if dangerous_arrow["exists"] == 1:
      agent_pos = np.array([my_pos['x'], my_pos['y']])
      arrow_pos = np.array([dangerous_arrow['x'], dangerous_arrow['y']])
      arrow_vel = np.array([dangerous_arrow['vx'], dangerous_arrow['vy']])
      if np.dot((agent_pos - arrow_pos), arrow_vel) > 0 and np.linalg.norm((agent_pos - arrow_pos)) < 40: #if the arrow is actually coming toward us
        self.button_pressed = 'z'

    # Presses jump in 1/20 of the frames.
    if random.randint(0, 10) == 0:
      self.button_pressed = 'j'

    return np.array([
      self.dir_to_input.index(self.direction_pressed), 
      self.but_to_input.index(self.button_pressed)
    ])
