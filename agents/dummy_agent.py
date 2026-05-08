import logging
import random
from typing import Any, Mapping

from towerfall import Connection

class DummyAgent:
  '''
  A minimal agent that works with the PettingZoo Env.

  params connection: A connection to a Towerfall game.
  params attack_archers: If True, the agent will attack other neutral archers.
  '''
  def __init__(self, connection: Connection, attack_archers: bool = False):
    self.state_init: Mapping[str, Any] = {}
    self.state_scenario: Mapping[str, Any] = {}
    self.state_update: Mapping[str, Any] = {}
    self.pressed = set()
    self.connection = connection
    self.attack_archers = attack_archers
    self.pos: Mapping[str, float] | None = None
    self.vel: Mapping[str, float] | None = None
    self.dead: bool = False
    self.arrows: int = 0

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
    #self.pos = None
    #self.vel = None
    self.dead = False
    self.arrows = 0
    for e in game_state['entities']:
      if e['type'] == 'archer' and e['playerIndex'] == self.state_init['index']:
        self.pos = e['pos']
        self.vel = e['vel']
        self.dead = e.get('dead', False)
        self.arrows = len(e.get('arrows', []))
        return
    # Archer wasn't in the entity list — treat as dead.
    self.dead = True

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
