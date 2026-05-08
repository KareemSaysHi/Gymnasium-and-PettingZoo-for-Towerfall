from agents import SimpleAgent
from common.logging_options import default_logging
from towerfall import Towerfall

default_logging()

'''
Three agents playing a 2 versus 2 match alongside a human.
BUG: only the first slot can be human.
'''

def main():
  # Creates or reuse a Towerfall game.
  towerfall = Towerfall(
    verbose = 0,
    config = dict(
      mode='sandbox',
      level='1',
      fps=60,
      solids=[[1] + [0] * 30 + [1]] * 14 + [[1] * 32] + [[1] + [0] * 30 + [1]] * 9,
      agentTimeout='00:00:02',
      agents=[
        dict(type='human', archer='yellow', team='blue'),
        dict(type='remote', archer='orange', team='red')]
    )
  )

  connections = []
  agents = []
  remote_agents = sum(1 for agent in towerfall.config['agents'] if agent['type'] != 'human')
  for i in range(remote_agents):
    connections.append(towerfall.join(timeout=20, verbose=0))
    agents.append(SimpleAgent(connections[i]))

  while True:
    # Read the state of the game then replies with an action.
    for connection, agent in zip(connections, agents):
      game_state = connection.read_json()
      print(f"game state: {game_state}")
      agent.act(game_state)


if __name__ == '__main__':
  main()