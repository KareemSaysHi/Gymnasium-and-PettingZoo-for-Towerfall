import re
import matplotlib.pyplot as plt

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
