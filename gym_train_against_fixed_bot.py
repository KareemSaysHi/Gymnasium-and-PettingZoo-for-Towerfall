from towerfall_env_gym import TowerFallEnvGym
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from agents import SmarterBaselineAgent
import os
os.makedirs("./gym_training_data", exist_ok=True)

env = DummyVecEnv([lambda: TowerFallEnvGym(opponent_bot=SmarterBaselineAgent())])

my_list = ["my_movement", "your_movement"]
for i in range (6):
    my_list.append(f"arrow{i}_pos-vel")

env = VecNormalize(env, norm_obs_keys= my_list, norm_reward = False)

model = MaskablePPO("MultiInputPolicy", env, verbose=1)

for i in range(100):
    #train 20k steps
    model.learn(total_timesteps=20000, reset_num_timesteps=False)
    model.save(f"./gym_training_data/versus_fixed_smarter{i*20000}")
    env.save(f"./gym_training_data/vn_versus_fixed_smarter{i*20000}.pkl")
