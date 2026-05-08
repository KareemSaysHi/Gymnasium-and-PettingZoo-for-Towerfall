from typing import Any, Mapping

from common.logging_options import default_logging
from towerfall import Towerfall

from agents import SimpleAgent, SimpleAgentNoShoot

default_logging()

'''
Two simple agents fighting each other in sandbox mode.
Immediately resets to the starting positions the instant either archer dies.
'''


def main():
  towerfall = Towerfall(
    verbose=1,
    config=dict(
      mode='sandbox',
      level='1',
      fps=60,
      agentTimeout='00:00:02',
      solids=[[1] + [0] * 30 + [1]] * 14 + [[1] * 32] + [[1] + [0] * 30 + [1]] * 9,
      agents=[
        dict(type='remote', archer='green'),
        dict(type='remote', archer='blue'),
      ],
    )
  )

  connections = [towerfall.join(timeout=10, verbose=1) for _ in range(2)]
  agents = [SimpleAgentNoShoot(c, attack_archers=True) for c in connections]

  send_reset(towerfall)

  while True:
    needs_reset = False
    for connection, agent in zip(connections, agents):
      game_state = connection.read_json()
      if not needs_reset and is_round_over(game_state):
        needs_reset = True
      agent.act(game_state)


    if game_state['type'] == 'update':
      for e in game_state['entities']:
          if e['type'] == 'archer':
              print(f"p{e['playerIndex']}: pos={e['pos']} vel={e['vel']}")
    if needs_reset:
      send_reset(towerfall)


def send_reset(towerfall: Towerfall):
  towerfall.send_reset([
    dict(type='archer', pos=dict(x=80, y=110)),
    dict(type='archer', pos=dict(x=240, y=110)),
  ])


def is_round_over(game_state: Mapping[str, Any]) -> bool:
  if game_state['type'] != 'update':
    return False
  archers_alive = sum(1 for e in game_state['entities'] if e['type'] == 'archer')
  return archers_alive < 2


if __name__ == '__main__':
  main()
