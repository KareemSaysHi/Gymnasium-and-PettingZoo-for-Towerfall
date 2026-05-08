from petting_zoo_env_shooting import TowerFallEnv
from agents import PPOComparisonAgent

# evaluate a model against SimpleAgent

num_rounds = 5
agents = ["archer_0", "archer_1"]
current_agents = {
    "archer_0": PPOComparisonAgent(), 
    "archer_1": PPOComparisonAgent()
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