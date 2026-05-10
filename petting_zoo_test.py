from towerfall_env_petting_zoo import TowerFallEnv
from agents import BaselineAgent
from agents import ModelAgent, SmarterBaselineAgent
import pickle
from sb3_contrib.ppo_mask import MaskablePPO

# Evaluate a model against BaselineAgent

model = MaskablePPO.load(f"./training_data/towerfall_masked_{600000}")
with open(f"./training_data/vn_masked_stats_{600000}.pkl", "rb") as f:
    normalizer = pickle.load(f)

num_rounds = 100
agents = ["archer_0", "archer_1"]
current_agents = {
    "archer_0": ModelAgent(model = model, normalizer = normalizer),  #if you don't have a model, replace with another BaselineAgent
    "archer_1": BaselineAgent()
}

#initialize environment
env = TowerFallEnv(fps=60)
obs, _ = env.reset()


while (num_rounds >= 0):
    action = {a: current_agents[a].take_action(obs[a]) for a in agents}
    obs, reward, term, trunc, _ = env.step(action)
    if term['archer_0'] or trunc['archer_0']:
        obs, info = env.reset()
        num_rounds -= 1