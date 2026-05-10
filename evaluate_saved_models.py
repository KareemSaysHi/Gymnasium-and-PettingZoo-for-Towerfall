from evaluate import evaluate
from sb3_contrib.ppo_mask import MaskablePPO
import pickle 
import re
import matplotlib.pyplot as plt

'''Evaluates a saved model against a fixed pre-scripted bot at various time steps
and produces a graph of the results '''
for i in range (0, 100):
    model = MaskablePPO.load(f"./training_data/towerfall_masked_{i*20000}")
    with open(f"./training_data/vn_masked_stats_{i*20000}.pkl", "rb") as f:
        normalizer = pickle.load(f)

    wins, losses, ties = evaluate(model, normalizer, 100)
    
    with open("data.txt", "a") as f:
       f.write("\n" + f"wins for {20*(i+1)}k: {wins}, losses for {20*(i+1)}k: {losses}")

LINE_RE = re.compile(r"wins for (\d+)k:\s*(\d+),\s*losses for \d+k:\s*(\d+)")

steps, wins, losses = [], [], []
with open("data.txt") as f:
    for line in f:
        m = LINE_RE.search(line)
        if not m:
            continue
        k, w, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
        steps.append(k)
        wins.append(w)
        losses.append(l)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(steps, wins,                color="green", width=15, label="Wins")
ax.bar(steps, losses, bottom=wins, color="red",   width=15, label="Losses")

ax.set_xlabel("Training steps (k)")
ax.set_ylabel("Count out of 100")
ax.set_title("Eval results vs. training steps")
ax.set_ylim(0, 100)
ax.legend(loc="upper right")
ax.set_xticks(steps[::2])
plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()
