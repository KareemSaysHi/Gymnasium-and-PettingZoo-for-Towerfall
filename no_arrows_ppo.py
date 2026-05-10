'''
Training the no-arrows environment for testing before moving to the
more general setting of fighting with arrows.

Note that we vectorize the towerfall petting zoo environment, so 
ONE MODEL is controlling both agents.  The observations of both 
players are used to train the model (so it learns from both sides).
'''

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import supersuit as ss
from no_arrows_env import TowerFallNoArrows 

#make a gym environment from a pettingzoo environment by vectorizing the agents
env0 = ss.pettingzoo_env_to_vec_env_v1(TowerFallNoArrows(fps=10000)) 
env = ss.concat_vec_envs_v1(env0, num_vec_envs=1, num_cpus=4, base_class="stable_baselines3")

env = VecNormalize(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=300000)
model.save("towerfall_no_arrows_normalized_300k")

