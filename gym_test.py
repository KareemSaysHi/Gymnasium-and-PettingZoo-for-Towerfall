from towerfall_env_gym import TowerFallEnvGym
from agents import BaselineAgent

# have two Baseline Agents play 5 rounds together through the Gym environment

num_rounds = 5


#initialize environment
env = TowerFallEnvGym(fps=60)
obs, _ = env.reset()
print(obs)

my_agent = BaselineAgent()

while (num_rounds > 0):
    action = my_agent.take_action(obs)
    obs, reward, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, info = env.reset()
        num_rounds -= 1