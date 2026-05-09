'''
Petting Zoo Vectorized PPO:

We vectorize the towerfall petting zoo environment, so ONE MODEL is controlling both agents.  The observations of both players are used to train the model (so it learns from both sides).
'''
from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
import supersuit as ss
from towerfall_env_petting_zoo import TowerFallEnv 
from supersuit.vector.concat_vec_env import ConcatVecEnv
import pickle
from evaluate import evaluate

#some janky Claude workaround for making sure all the correct functions are in ConcatVecEnv
def _has_attr(self, attr_name):
    return all(hasattr(v, attr_name) for v in self.vec_envs)

def _env_method(self, method_name, *args, indices=None, **kwargs):
    results = []
    for vec_env in self.vec_envs:
        method = getattr(vec_env, method_name)
        result = method(*args, **kwargs)
        # If the method returns one entry per sub-env, flatten it.
        # Otherwise broadcast the single result across this vec_env's sub-envs.
        if hasattr(result, "__len__") and not isinstance(result, str) and len(result) == vec_env.num_envs:
            results.extend(list(result))
        else:
            results.extend([result] * vec_env.num_envs)
    if indices is None:
        return results
    return [results[i] for i in indices]

def _get_attr(self, attr_name, indices=None):
    results = []
    for vec_env in self.vec_envs:
        attr = getattr(vec_env, attr_name)
        results.extend([attr] * vec_env.num_envs)
    if indices is None:
        return results
    return [results[i] for i in indices]

def _env_is_wrapped(self, wrapper_class, indices=None):
    return [False] * self.num_envs   # SuperSuit doesn't track this; safe default

ConcatVecEnv.has_attr = _has_attr
ConcatVecEnv.env_method = _env_method
ConcatVecEnv.get_attr = _get_attr
ConcatVecEnv.env_is_wrapped = _env_is_wrapped
#end of janky Claude workaround

# Helper wrapper that is necessary for MaskablePPO on a vectorized environment:
class MarkovVectorEnvActionMasker(ss.vector.markov_vector_wrapper.MarkovVectorEnv):
    def action_masks(self):
        agent_action_masks = self.par_env.action_masks()
        action_masks_list = np.stack([agent_action_masks[a] for a in self.par_env.agents]) #technically should be possible_agents but the number of agents never change
        return action_masks_list
        
#make a gym environment from a pettingzoo environment by vectorizing the agents
env0 = MarkovVectorEnvActionMasker(TowerFallEnv(fps=1000)) 
env1 = ss.concat_vec_envs_v1(env0, num_vec_envs=1, base_class="stable_baselines3") #turn into sb3 object

my_list = ["my_movement", "your_movement"]
for i in range (6):
    my_list.append(f"arrow{i}_pos-vel")
env = VecNormalize(env1, norm_obs_keys= my_list, norm_reward = False)
evaluating_env = TowerFallEnv(fps = 10000)

model = MaskablePPO("MultiInputPolicy", env, verbose=1)
for i in range(100):
    #train 20k steps
    model.learn(total_timesteps=20000, reset_num_timesteps=False)
    model.save(f"./training_data/towerfall_masked_{i*20000}")
    env.save(f"./training_data/vn_masked_stats_{i*20000}.pkl")
    env.reset()

    #evaluating
    #with open("./training_data/vn_masked_stats.pkl", "rb") as f:
    #    normalizer = pickle.load(f)
    # 
    #wins, losses, ties = evaluate(model, normalizer, evaluating_env, 100)
    # 
    #with open("data.txt", "a") as f:
    #    f.write("\n" + f"wins for {20*(i+1)}k: {wins}, losses for {20*(i+1)}k: {losses}")
    
    #env.reset()





