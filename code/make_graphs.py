"""Result graphs for the report (reads results/results.csv, results/dataset_distribution.csv).
python ../code/make_graphs.py   ->  figures/graph_*.png
Colours: validated categorical slots 1-3 of the reference palette (blue, orange, aqua); thin recessive grid; values labelled."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figures", exist_ok=True)
res = pd.read_csv("results/results.csv")
res = res[~res.tag.str.contains("_lr")]
NAMES = {"baseline": "LeafNet baseline (scratch)", "A_se": "LeafNet + SE (A)", "AB_se_bgaug": "LeafNet + A + B",
         "ABC_proposed": "LeafNet A+B+C (scratch)", "ref_resnet": "ResNet-style (scratch)", "ref_mobilenet": "MobileNet-style (scratch)",
         "ref_tinyvit": "Tiny ViT (scratch)", "ref_pre_mnv3": "MobileNetV3-L pretrained", "ref_pre_effb0": "EfficientNet-B0 pretrained",
         "pre_mnv3_bgaug": "MobileNetV3-L pretrained + B"}
agg = res.groupby("variant").agg(acc=("accuracy", "mean"), acc_sd=("accuracy", lambda x: x.std(ddof=0)),
                                 bg=("bgswap_accuracy", "mean"), bg_sd=("bgswap_accuracy", lambda x: x.std(ddof=0)),
                                 params=("params", "first"), n=("accuracy", "size"))

def clean(ax):
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)

# 1. dataset distribution -------------------------------------------------------------------
d = pd.read_csv("results/dataset_distribution.csv").sort_values("full_dataset_images")
fig, ax = plt.subplots(figsize=(9, 9))
ax.barh(range(len(d)), d.full_dataset_images, color=BLUE, height=0.7)
cap = int(d[["train", "val", "test"]].sum(axis=1).max())
ax.axvline(cap, color=ORANGE, lw=1.2); ax.set_ylim(-2.0, len(d) - 0.4); ax.text(cap + 80, -1.2, f"orange line: cap used in the CPU study ({cap} images / class)", color=INK2, fontsize=8.5, va="center")
ax.set_yticks(range(len(d))); ax.set_yticklabels([c.replace("___", ": ").replace("_", " ")[:34] for c in d["class"]], fontsize=7)
for i, v in enumerate(d.full_dataset_images): ax.text(v + 40, i, f"{v:,}", va="center", fontsize=7, color=INK2)
ax.set_xlabel("images in full PlantVillage (54,305 total)"); ax.set_title("Class distribution: 36:1 imbalance between largest and smallest class", loc="left", fontsize=11)
clean(ax); fig.tight_layout(); fig.savefig("figures/graph_dataset_distribution.png", dpi=150); plt.close(fig)

# 2. clean vs background-swap accuracy per model -------------------------------------------------
order = ["ref_pre_mnv3", "ref_pre_effb0", "pre_mnv3_bgaug", "ref_mobilenet", "ref_resnet", "baseline", "ABC_proposed", "ref_tinyvit"]
a = agg.loc[[o for o in order if o in agg.index]].sort_values("acc")
fig, ax = plt.subplots(figsize=(10, 5.6)); y = np.arange(len(a)); h = 0.36
ax.barh(y + h / 2, a.acc, h - 0.04, color=BLUE, label="clean test images")
ax.barh(y - h / 2, a.bg, h - 0.04, color=ORANGE, label="background replaced (leaf unchanged)")
for i, (ac, bg) in enumerate(zip(a.acc, a.bg)):
    ax.text(ac + 0.008, i + h / 2, f"{ac:.3f}", va="center", fontsize=8, color=INK2); ax.text(bg + 0.008, i - h / 2, f"{bg:.3f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([NAMES[v] + (f"  (n={int(n)})" if n > 1 else "") for v, n in zip(a.index, a.n)], fontsize=8.5)
ax.set_xlim(0, 1.08); ax.set_xlabel("test accuracy"); clean(ax)
ax.set_title("Only background-randomised models survive a background swap", loc="left", fontsize=11)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5); fig.tight_layout(); fig.savefig("figures/graph_model_comparison.png", dpi=150); plt.close(fig)

# 3. ablation (two panels, mean +- std over seeds) ----------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [4, 2.4]}, sharey=True)
for ax, vs, ttl in [(axs[0], ["baseline", "A_se", "AB_se_bgaug", "ABC_proposed"], "From-scratch LeafNet: components added one by one"),
                    (axs[1], ["ref_pre_mnv3", "pre_mnv3_bgaug"], "MobileNetV3-L (ImageNet-pretrained)")]:
    vs = [v for v in vs if v in agg.index]; x = np.arange(len(vs)); w = 0.36
    for k, (col, sd, c, lab) in enumerate([("acc", "acc_sd", BLUE, "clean accuracy"), ("bg", "bg_sd", ORANGE, "background-swap accuracy")]):
        vals, errs = agg.loc[vs, col].values, agg.loc[vs, sd].values
        ax.bar(x + (k - 0.5) * w, vals, w - 0.04, color=c, label=lab, yerr=errs, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for xi, v in zip(x + (k - 0.5) * w, vals): ax.text(xi, v + 0.035, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
    SHORT = {"baseline": "baseline", "A_se": "+ SE (A)", "AB_se_bgaug": "+ A + B", "ABC_proposed": "+ A + B + C",
             "ref_pre_mnv3": "plain fine-tuning", "pre_mnv3_bgaug": "+ B (background\nrandomisation)"}
    ax.set_xticks(x); ax.set_xticklabels([SHORT[v] + f"\n(n={int(agg.loc[v, 'n'])})" for v in vs], fontsize=8)
    ax.set_title(ttl, loc="left", fontsize=10); ax.set_ylim(0, 1.12); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
axs[0].set_ylabel("test accuracy (mean, bars = std over seeds)"); axs[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
fig.suptitle("Ablation: background randomisation (B) trades a little clean accuracy for large robustness", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig("figures/graph_ablation.png", dpi=150); plt.close(fig)

# 4. accuracy vs parameters ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.4))
pre = {"ref_pre_mnv3", "ref_pre_effb0", "pre_mnv3_bgaug"}
for v, r in agg.iterrows():
    c = ORANGE if v in pre else BLUE
    ax.scatter(r.params / 1e6, r.acc, s=46, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
    off, ha = {"ref_pre_mnv3": ((-9, 11), "right"), "ref_pre_effb0": ((-9, -1), "right"), "pre_mnv3_bgaug": ((-9, -14), "right")}.get(v, ((6, 4), "left"))
    ax.annotate(NAMES[v], (r.params / 1e6, r.acc), xytext=off, textcoords="offset points", fontsize=7.5, color=INK2, ha=ha)
ax.set_xscale("log"); ax.set_xticks([0.35, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.35", "0.5", "1", "2", "4"]); ax.minorticks_off()
ax.set_xlabel("parameters (millions, log scale)"); ax.set_ylabel("clean test accuracy")
ax.scatter([], [], color=BLUE, label="trained from scratch"); ax.scatter([], [], color=ORANGE, label="ImageNet-pretrained, fine-tuned")
ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("Pretraining, not size, drives accuracy (96 px, 7,552-image subset)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("figures/graph_accuracy_vs_params.png", dpi=150); plt.close(fig)

# 5. pipeline diagram ---------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 3.6)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 3.6)
boxes = [("PlantVillage\n54,305 images\n38 classes\n(colour + segmented)", 0.1), ("Preprocessing\npair colour/segmented\nleaf mask, 96 px\nstratified 70/15/15", 2.7),
         ("Models\nLeafNet (A,B,C)\nResNet/MobileNet/ViT\nMobileNetV3, EffNet-B0\n(ImageNet-pretrained)", 5.3), ("Training\nAdamW + OneCycle\nbackground\nrandomisation (B)", 7.9),
         ("Evaluation (test)\nacc/P/R/F1/spec/AUC/ECE\nbackground-swap test\nGrad-CAM leaf focus", 10.5)]
for t, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.7), 2.3, 2.2, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=BLUE, lw=1.2))
    ax.text(x + 1.15, 1.8, t, ha="center", va="center", fontsize=8.5, color=INK)
for x in [2.45, 5.05, 7.65, 10.25]: ax.annotate("", xy=(x + 0.22, 1.8), xytext=(x - 0.02, 1.8), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
ax.text(0.1, 0.25, "CO3 implementation  ->  CO4 comparison on the same split  ->  CO5 ablation of the background-robust training", fontsize=9, color=INK2)
fig.savefig("figures/graph_pipeline.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("graphs written:", sorted(f for f in os.listdir("figures") if f.startswith("graph_")))
