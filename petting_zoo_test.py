from petting_zoo_env import TowerFallNoArrows 


env = TowerFallNoArrows(fps=60)
obs, info = env.reset()
for i in range (0, 10000):
    action = {a: env.action_space(a).sample() for a in env.agents}
    obs, reward, term, trunc, info = env.step(action)
    #print(term)
    if term['archer_0'] or trunc['archer_0']:
        obs, info = env.reset()
        
env.close()