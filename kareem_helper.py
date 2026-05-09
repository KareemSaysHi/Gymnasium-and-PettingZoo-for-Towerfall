

from towerfall_env_gym import TowerFallEnvGym
from agents import ModelAgent, SmarterBaselineAgent
import pickle
from sb3_contrib.ppo_mask import MaskablePPO
# evaluate a model against SimpleAgent


model = MaskablePPO.load(f"./gym_training_data/versus_fixed_{260000}")
with open(f"./gym_training_data/vn_versus_fixed_{260000}.pkl", "rb") as f:
    normalizer = pickle.load(f)
num_rounds = 10


#initialize environment
env = TowerFallEnvGym(fps=60, opponent_bot=SmarterBaselineAgent())
obs, _ = env.reset()
print(obs)

my_agent = ModelAgent(model = model, normalizer= normalizer)

while (num_rounds > 0):
    action = my_agent.take_action(obs)
    obs, reward, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, info = env.reset()
        num_rounds -= 1