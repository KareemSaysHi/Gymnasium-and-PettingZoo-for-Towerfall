from .petting_zoo_env_shooting import TowerFallEnv
from agents import PPOComparisonAgent

# evaluate a model against SimpleAgent

num_episodes = 200

#initialize environment
env = TowerFallEnv(fps=1000)
obs, _ = env.reset()


#then just run the guy