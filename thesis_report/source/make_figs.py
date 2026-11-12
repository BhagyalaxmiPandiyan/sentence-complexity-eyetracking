import sys, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
out = sys.argv[1]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]

# 3.1 overview
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=0.8))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=9.5)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1))
ax.text(0.1, 4.75, "English", fontsize=11, weight="bold"); ax.text(0.1, 2.25, "Tamil", fontsize=11, weight="bold")
box(0.1, 3.3, 2.0, 1.1, "Dundee eye-tracking\n+ lexical / syntactic\nfeatures", "#E8EEF7")
box(2.7, 3.3, 2.1, 1.1, "System 1\nword reading times\n(FFD, FPD, RPD, TD)", "#E8EEF7")
box(5.4, 3.3, 2.0, 1.1, "System 2\nwhich sentence\nis harder?", "#E8EEF7")
box(2.7, 4.5, 2.1, 0.45, "Phase 2: + GPT-2 surprisal", "#FBE9DD")
box(8.0, 3.3, 1.9, 1.1, "Toolkit\n(live demo)", "#E6F2E8")
arr(2.1, 3.85, 2.7, 3.85); arr(4.8, 3.85, 5.4, 3.85); arr(7.4, 3.85, 8.0, 3.85); arr(3.75, 4.5, 3.75, 4.42)
box(0.1, 0.8, 2.0, 1.1, "Graded text\n(textbooks,\nStoryWeaver)", "#F3EAF5")
box(2.7, 0.8, 2.1, 1.1, "Word metric (A)\nvs morph metric\n(A + B)", "#F3EAF5")
box(5.4, 0.15, 2.0, 0.95, "LLM simplification\n+ human ratings", "#F3EAF5")
box(5.4, 1.35, 2.0, 0.95, "Webcam\neye-tracking", "#F3EAF5")
arr(2.1, 1.35, 2.7, 1.35); arr(4.8, 1.55, 5.4, 1.8); arr(4.8, 1.15, 5.4, 0.65); arr(7.4, 1.35, 8.4, 3.3)
fig.tight_layout(); fig.savefig(f"{out}/fig3_1_overview.png", dpi=220); plt.close()

# 4.1 System 1 R2
m = ["FFD", "FPD", "RPD", "TD"]
base = [0.447, 0.561, 0.558, 0.538]; gpt = [0.458, 0.583, 0.582, 0.568]; paper = [0.649, 0.600, 0.570, 0.516]
fig, ax = plt.subplots(figsize=(6.2, 3.1)); w = 0.26
for i, (v, lab) in enumerate([(base, "This work (replication)"), (gpt, "This work + GPT-2 surprisal"), (paper, "Singh et al. (2016)")]):
    b = ax.bar([j + (i - 1) * w for j in range(4)], v, w, label=lab, color=C[i])
    ax.bar_label(b, fmt="%.3f", fontsize=7.5, padding=1)
ax.set_xticks(range(4), m); ax.set_ylabel("Test R²"); ax.set_ylim(0, 0.75); ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); fig.savefig(f"{out}/fig4_1_system1.png", dpi=220); plt.close()

# 4.2 ablation
steps = ["Word length", "+ Sent. length", "+ Frequency", "+ Mean AoA", "+ SD AoA", "+ Transition prob.", "+ Lexical surp.", "+ Syntactic surp.", "+ Entropy red.", "+ Emb. depth", "+ Emb. difference", "+ h1–h8"]
ours = [0.447, 0.447, 0.505, 0.505, 0.505, 0.506, 0.506, 0.506, 0.506, 0.506, 0.507, 0.506]
pap = [0.267, 0.500, 0.506, 0.510, 0.516, 0.544, 0.568, 0.575, 0.579, 0.580, 0.581, 0.585]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(range(12), ours, "o-", color=C[0], label="This work"); ax.plot(range(12), pap, "s--", color=C[2], label="Singh et al. (2016)")
ax.set_xticks(range(12), steps, rotation=40, ha="right", fontsize=8.5); ax.set_ylabel("Cumulative dev R² (FFD)"); ax.set_ylim(0.2, 0.65); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{out}/fig4_2_ablation.png", dpi=220); plt.close()

# 5.2 eye-tracking R2
groups = ["Controls only", "A: word-level", "B: morphology", "A + B"]
vals = {"FFD": [.100, .105, .099, .103], "FPD": [.087, .141, .120, .137], "RPD": [.037, .132, .094, .130], "TD": [.027, .103, .071, .099]}
fig, ax = plt.subplots(figsize=(6.2, 3.0)); w = 0.2
for i, g in enumerate(groups):
    ax.bar([j + (i - 1.5) * w for j in range(4)], [vals[k][i] for k in m], w, label=g, color=C[i])
ax.set_xticks(range(4), m); ax.set_ylabel("Cross-validated R²"); ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); fig.savefig(f"{out}/fig5_2_eyetracking.png", dpi=220); plt.close()

# 5.3 bands
fig, ax = plt.subplots(figsize=(4.2, 2.7))
b = ax.bar(["Easy", "Medium", "Hard"], [7.2, 9.4, 11.9], color=[C[2], C[1], C[3]], width=0.55); ax.bar_label(b, fmt="%.1f s", fontsize=9)
ax.set_ylabel("Mean sentence reading time (s)"); ax.set_ylim(0, 14)
fig.tight_layout(); fig.savefig(f"{out}/fig5_3_bands.png", dpi=220); plt.close()
print("ok")
