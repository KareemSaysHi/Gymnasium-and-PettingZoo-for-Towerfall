''' TowerFallEnvGym is built on top of the corresponding Petting Zoo environment.  Go check that file out first for more information.'''


import gymnasium as gym
from agents import PPOComparisonAgent
from towerfall_env_petting_zoo import TowerFallEnv

class TowerFallEnvGym(gym.Env):

    def __init__(self, fps = 10000, frames_per_action = 5, opponent_bot = PPOComparisonAgent()):
        #make secret internal PettingZooEnv
        self.fps = fps
        self.frames_per_action = frames_per_action
        self.petting_zoo_env = TowerFallEnv(fps = self.fps, frames_per_action = self.frames_per_action)

        #set observation and action spaces
        self.observation_space = self.petting_zoo_env.observation_space("archer_0")
        self.action_space = self.petting_zoo_env.action_space("archer_0")
    
        self.opponent_bot = opponent_bot
        self.opponent_bot_obs = None

        
    def action_masks(self):
        return self.petting_zoo_env.action_masks()["archer_0"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) #necessary for gymnasium
        obs, info = self.petting_zoo_env.reset()
        self.opponent_bot_obs = obs['archer_1'] 
        return obs['archer_0'], info['archer_0']

    def step(self, action):
        
        petting_zoo_action = {
            "archer_0": action,
            "archer_1": self.opponent_bot.take_action(self.opponent_bot_obs)
        }

        observations, rewards, terminations, truncations, infos = self.petting_zoo_env.step(petting_zoo_action)
        self.opponent_bot_obs = observations['archer_1'] 
        return observations["archer_0"], rewards["archer_0"], terminations["archer_0"], truncations["archer_0"], infos["archer_0"]
    
