from towerfall_env_petting_zoo import TowerFallEnv
from agents import PPOComparisonAgent, ModelAgent
from stable_baselines3 import PPO
import pickle

# evaluation of one model against another

def evaluate(model, normalizer, num_rounds):

    agents = ["archer_0", "archer_1"]

    current_agents = {
        "archer_0": ModelAgent(model = model, normalizer = normalizer), 
        "archer_1": PPOComparisonAgent()
    }

    first_player_wins = 0
    first_player_losses = 0
    first_player_ties = 0

    #initialize environment
    env = TowerFallEnv(fps=1000)
    obs, _ = env.reset()


    while (num_rounds >= 0):
        action = {a: current_agents[a].take_action(obs[a]) for a in agents}
        obs, reward, term, trunc, info = env.step(action)
        if term['archer_0'] or trunc['archer_0']:
            if info["archer_0"]["dead"] and not info["archer_1"]["dead"]:
                first_player_losses += 1
            elif not info["archer_0"]["dead"] and info["archer_1"]["dead"]:
                first_player_wins += 1
            else:
                first_player_ties += 1

            obs, info = env.reset()
            num_rounds -= 1

    return [first_player_wins, first_player_losses, first_player_ties]
