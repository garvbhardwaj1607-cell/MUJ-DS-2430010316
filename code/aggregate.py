"""Aggregate metrics into the tables/figures required by the assignment.
python ../code/aggregate.py
Writes: results/results.csv, results/ablation.csv, results/comparison.csv, results/lr_sensitivity.csv,
        figures/training_curve.png, figures/learning_curves_ablation.png, figures/gradcam_examples.png
"""
import glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import VARIANTS
from utils import load_data, swap_background, grad_cam

ABL = ["baseline", "A_se", "AB_se_bgaug", "ABC_proposed"]
REF = ["ref_resnet", "ref_mobilenet", "ref_tinyvit", "ref_pre_mnv3", "ref_pre_effb0"]
COMP = {"baseline": (0, 0, 0), "A_se": (1, 0, 0), "AB_se_bgaug": (1, 1, 0), "ABC_proposed": (1, 1, 1)}
NAMES = {"baseline": "Baseline LeafNet", "A_se": "+ SE attention (A)", "AB_se_bgaug": "+ background randomisation (A+B)",
         "ABC_proposed": "+ mask-supervised spatial attention (A+B+C) = LeafNet A+B+C",
         "ref_resnet": "ResNet-style (scratch)", "ref_mobilenet": "MobileNet-style (scratch)", "ref_tinyvit": "Tiny ViT (scratch)",
         "ref_pre_mnv3": "MobileNetV3-L (ImageNet-pretrained, fine-tuned)", "ref_pre_effb0": "EfficientNet-B0 (ImageNet-pretrained, fine-tuned)",
         "pre_mnv3_bgaug": "MobileNetV3-L pretrained + background randomisation (B) = Proposed"}


def ms(x, d=4):
    x = pd.Series(x).dropna()
    if len(x) == 0: return ""
    return f"{x.mean():.{d}f} ± {x.std(ddof=0):.{d}f}" if len(x) > 1 else f"{x.iloc[0]:.{d}f}"


def main():
    rows = [json.load(open(f)) for f in sorted(glob.glob("results/metrics_*.json"))]
    df = pd.DataFrame(rows)
    df.to_csv("results/results.csv", index=False)
    main_df = df[~df.tag.str.contains("_lr")]

    # ---- ablation (CO5)
    ab = []
    for v in ABL:
        d = main_df[main_df.variant == v]
        if d.empty: continue
        ab.append({"Experiment": NAMES[v], "A: SE": "✓" if COMP[v][0] else "✗", "B: BG-rand": "✓" if COMP[v][1] else "✗",
                   "C: mask-attn": "✓" if COMP[v][2] else "✗", "seeds": len(d),
                   "Accuracy": ms(d.accuracy), "Macro-F1": ms(d.f1_macro), "BG-swap Acc": ms(d.bgswap_accuracy),
                   "BG-swap F1": ms(d.bgswap_f1_macro), "ECE": ms(d.ece), "Leaf-focus (clean)": ms(d.leaf_focus_clean),
                   "Leaf-focus (BG-swap)": ms(d.leaf_focus_bgswap), "Deletion gain (clean)": ms(d.deletion_gain_clean),
                   "Params": int(d.params.iloc[0])})
    pd.DataFrame(ab).to_csv("results/ablation.csv", index=False)

    # ---- ablation on the pretrained backbone (CO5): plain fine-tuning vs + background randomisation (B)
    pa = []
    for v, lab, bflag in [("ref_pre_mnv3", "MobileNetV3-L pretrained (plain)", "✗"),
                          ("pre_mnv3_bgaug", "MobileNetV3-L pretrained + background randomisation (B)", "✓")]:
        d = main_df[main_df.variant == v]
        if d.empty: continue
        pa.append({"Experiment": lab, "B: BG-rand": bflag, "seeds": len(d), "Accuracy": ms(d.accuracy), "Macro-F1": ms(d.f1_macro),
                   "BG-swap Acc": ms(d.bgswap_accuracy), "BG-swap F1": ms(d.bgswap_f1_macro), "ECE": ms(d.ece),
                   "Leaf-focus (clean)": ms(d.leaf_focus_clean), "Leaf-focus (BG-swap)": ms(d.leaf_focus_bgswap),
                   "Params": int(d.params.iloc[0])})
    pd.DataFrame(pa).to_csv("results/ablation_pretrained.csv", index=False)

    # ---- comparison (CO4): references trained under identical split/budget
    cp = []
    for v in REF + ["baseline", "ABC_proposed", "pre_mnv3_bgaug"]:
        d = main_df[main_df.variant == v]
        if d.empty: continue
        cp.append({"Model": NAMES[v], "seeds": len(d), "Accuracy": ms(d.accuracy), "Precision(macro)": ms(d.precision_macro),
                   "Recall(macro)": ms(d.recall_macro), "F1(macro)": ms(d.f1_macro), "AUC(macro OvR)": ms(d.roc_auc_macro_ovr, 5),
                   "BG-swap Acc": ms(d.bgswap_accuracy), "Params": int(d.params.iloc[0]),
                   "CPU ms/img (1 thread)": ms(d.latency_ms_cpu_1thread, 2)})
    pd.DataFrame(cp).to_csv("results/comparison.csv", index=False)

    # ---- learning-rate sensitivity (baseline, seed 0, validation accuracy)
    lr = []
    for f in sorted(glob.glob("results/train_baseline_s0*.json")):
        j = json.load(open(f))
        lr.append({"run": j["tag"], "learning_rate": j["lr"], "best_val_acc": round(j["best_val_acc"], 4),
                   "train_seconds": round(j["train_seconds"])})
    pd.DataFrame(lr).to_csv("results/lr_sensitivity.csv", index=False)

    # ---- training curves
    os.makedirs("figures", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for tag, c in [("baseline_s0", "tab:gray"), ("ABC_proposed_s0", "tab:red")]:
        f = f"results/history_{tag}.csv"
        if not os.path.exists(f): continue
        h = pd.read_csv(f)
        axs[0].plot(h.epoch, h.train_loss, "--", c=c, label=f"{tag} train"); axs[0].plot(h.epoch, h.val_loss, "-", c=c, label=f"{tag} val")
        axs[1].plot(h.epoch, h.train_acc, "--", c=c, label=f"{tag} train"); axs[1].plot(h.epoch, h.val_acc, "-", c=c, label=f"{tag} val")
    axs[0].set_title("Loss"); axs[1].set_title("Accuracy")
    for a in axs: a.set_xlabel("epoch"); a.legend(fontsize=7); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figures/training_curve.png", dpi=150); fig.savefig("results/training_curve.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for tag, c in [("ref_pre_mnv3_s0", "tab:gray"), ("pre_mnv3_bgaug_s0", "tab:red")]:
        f = f"results/history_{tag}.csv"
        if not os.path.exists(f): continue
        h = pd.read_csv(f)
        axs[0].plot(h.epoch, h.train_loss, "--", c=c, label=f"{tag} train"); axs[0].plot(h.epoch, h.val_loss, "-", c=c, label=f"{tag} val")
        axs[1].plot(h.epoch, h.train_acc, "--", c=c, label=f"{tag} train"); axs[1].plot(h.epoch, h.val_acc, "-", c=c, label=f"{tag} val")
    axs[0].set_title("Loss"); axs[1].set_title("Accuracy")
    for a in axs: a.set_xlabel("epoch"); a.legend(fontsize=7); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figures/training_curve_pretrained.png", dpi=150); fig.savefig("results/training_curve_pretrained.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for v in ABL + REF:
        f = f"results/history_{v}_s0.csv"
        if os.path.exists(f):
            h = pd.read_csv(f); ax[0].plot(h.epoch, h.val_acc, label=NAMES[v][:34]); ax[1].plot(h.epoch, h.val_loss, label=NAMES[v][:34])
    ax[0].set_title("Validation accuracy (seed 0)"); ax[1].set_title("Validation loss (seed 0)")
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig("figures/learning_curves_ablation.png", dpi=150); plt.close(fig)

    # ---- qualitative Grad-CAM (clean vs background-swapped)
    data, classes = load_data()
    Xte, Mte, yte = data["test"]
    rng = np.random.RandomState(3); idx = torch.from_numpy(rng.choice(len(yte), 6, replace=False))
    x01 = Xte[idx].float() / 255
    xs = swap_background(x01, Mte[idx], torch.Generator().manual_seed(7), p=1.0)
    for GT, GF in [(["baseline_s0", "ABC_proposed_s0"], "figures/gradcam_examples.png"),
                   (["ref_pre_mnv3_s0", "pre_mnv3_bgaug_s0"], "figures/gradcam_pretrained.png")]:
        models = {}
        for t in GT:
            if os.path.exists(f"models/{t}.pt"):
                ck = torch.load(f"models/{t}.pt", weights_only=False)
                m = VARIANTS[ck["variant"]]["model"](); m.load_state_dict(ck["state"]); m.eval()
                models[t] = (m, ck["mean"], ck["std"])
        if models:
            fig, axs = plt.subplots(1 + 2 * len(models), 6, figsize=(12, 2 * (1 + 2 * len(models))))
            for j in range(6):
                axs[0, j].imshow(x01[j].permute(1, 2, 0)); axs[0, j].set_title(classes[int(yte[idx][j])].split("___")[1][:16], fontsize=7)
            row = 1
            for t, (m, mean, std) in models.items():
                for lab, imgs in [("clean", x01), ("BG-swap", xs)]:
                    cam, lg = grad_cam(m, (imgs - mean) / std)
                    for j in range(6):
                        c = cam[j] / cam[j].max().clamp_min(1e-8)
                        axs[row, j].imshow(imgs[j].permute(1, 2, 0)); axs[row, j].imshow(c, cmap="jet", alpha=0.45)
                        ok = lg[j].argmax().item() == int(yte[idx][j])
                        axs[row, j].set_xlabel("correct" if ok else "wrong", fontsize=6)
                    axs[row, 0].set_ylabel(f"{t.split('_s')[0]}\n{lab}", fontsize=7)
                    row += 1
            for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
            fig.tight_layout(); fig.savefig(GF, dpi=150); plt.close(fig)
    print(pd.DataFrame(ab).to_string(index=False)); print(); print(pd.DataFrame(cp).to_string(index=False))


if __name__ == "__main__":
    main()
