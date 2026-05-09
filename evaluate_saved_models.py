from evaluation import evaluate
from sb3_contrib.ppo_mask import MaskablePPO
import pickle 

for i in range (0, 100):
    model = MaskablePPO.load(f"./training_data/towerfall_masked_{i*20000}")
    with open(f"./training_data/vn_masked_stats_{i*20000}.pkl", "rb") as f:
        normalizer = pickle.load(f)

    wins, losses, ties = evaluate(model, normalizer, 100)
    
    with open("data.txt", "a") as f:
       f.write("\n" + f"wins for {20*(i+1)}k: {wins}, losses for {20*(i+1)}k: {losses}")
    