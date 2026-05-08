import random
from typing import Any, Mapping
from towerfall import Connection

class PPOComparisonAgent:

  def __init__(self, connection: Connection, attack_archers: bool = True):
    self.state_init = {}
    self.state_scenario = {}
    self.state_update = {}
    self.pressed = set()
    self.connection = connection
    self.attack_archers = attack_archers
    self.direction_pressed = None
    self.button_pressed = None

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
      self.state_update = game_state
      print(game_state)
      print("-------------------------------------")

    # After receiving an 'update', your bot is expected to output string with the pressed buttons.
    # Each button is represented by a character:
    # r = right
    # l = left
    # u = up
    # d = down
    # j = jump
    # z = dash
    # s = shoot
    # The order of the characters are irrelevant. Any other character is ignored. Repeated characters are ignored.

    # This bot acts based on the position of the other player only. It
    # has a very random playstyle:
    #  - Runs to the enemy when they are below.
    #  - Runs away from the enemy when they are above.
    #  - Shoots when in the same horizontal line.
    #  - Dashes randomly.
    #  - Jumps randomly.

    my_state = None
    enemy_state = None

    players = []

    for state in self.state_update['entities']:
      if state['type'] == 'archer':
        players.append(state)
        if state['playerIndex'] == self.state_init['index']:
          my_state = state

    # Try to find an enemy archer.
    for state in players:
      if state['playerIndex'] == my_state['playerIndex']:
        continue
      if (self.attack_archers and state['team'] == 'neutral') or state['team'] != my_state['team']:
        enemy_state = state
        break

    # If no enemy archer is found, try to find another enemy.
    if not enemy_state:
      for state in self.state_update['entities']:
        if state['isEnemy']:
          enemy_state = state

    # If no enemy is found, means all are dead.
    if enemy_state == None:
      self.send_actions()
      return

    my_pos = my_state['pos']
    enemy_pos = enemy_state['pos']
    if enemy_pos['y'] >= my_pos['y'] and enemy_pos['y'] <= my_pos['y'] + 50:
      # Runs away if enemy is right above
      if my_pos['x'] < enemy_pos['x']:
        self.direction_pressed = 'l'
      else:
        self.direction_pressed = 'r'
    else:
      # Runs to enemy if they are below
      if my_pos['x'] < enemy_pos['x']:
        self.direction_pressed = 'r'
      else:
        self.direction_pressed = 'l'

      # If in the same line shoots,
      if abs(my_pos['y'] - enemy_pos['y']) < enemy_state['size']['y']:
        if random.randint(0, 1) == 0:
          self.button_pressed = 's'

    # Presses dash in 1/10 of the frames.
    if random.randint(0, 9) == 0:
      self.button_pressed = 'z'

    # Presses jump in 1/20 of the frames.
    if random.randint(0, 19) == 0:
      self.button_pressed = 'j'

    return self.direction_pressed, self.button_pressed

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
