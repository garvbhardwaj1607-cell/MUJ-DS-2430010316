"""Evaluate trained checkpoints on the held-out test split.

python ../code/test.py --tags baseline_s0 ABC_proposed_s0        (or --all)

Outputs per tag: results/metrics_<tag>.json, results/perclass_<tag>.csv, results/cm_<tag>.csv,
                 figures/confusion_<tag>.png
Metrics: accuracy, macro/weighted precision/recall/F1, macro specificity, macro one-vs-rest ROC-AUC, ECE,
         accuracy under background swap (robustness), CPU latency, and (CNNs) Grad-CAM leaf-focus + deletion AUC.
"""
import argparse, glob, json, os, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from model import VARIANTS, count_params
from utils import load_data, swap_background, grad_cam, leaf_focus, deletion_auc


@torch.no_grad()
def predict(model, x01, mean, std, bs=256):
    model.eval()
    outs = [F.softmax(model(((x01[i:i + bs]) - mean) / std), 1) for i in range(0, len(x01), bs)]
    return torch.cat(outs)


def ece_score(probs, y, bins=15):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.float().mean() * (acc[sel].mean() - conf[sel].mean()).abs()
    return float(e)


def plot_cm(cm, classes, path, title):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    short = [c.replace("___", ": ").replace("_", " ")[:28] for c in classes]
    ax.set_xticklabels(short, rotation=90, fontsize=5); ax.set_yticklabels(short, fontsize=5)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class"); ax.set_title(title)
    fig.colorbar(im, fraction=0.03, label="row-normalised"); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def evaluate_tag(tag, data, classes, n_xai=400):
    ck = torch.load(f"models/{tag}.pt", weights_only=False)
    model = VARIANTS[ck["variant"]]["model"](); model.load_state_dict(ck["state"]); model.eval()
    mean, std = ck["mean"], ck["std"]
    Xte, Mte, yte = data["test"]
    x01 = Xte.float() / 255
    probs = predict(model, x01, mean, std)
    pred = probs.argmax(1)
    y, p = yte.numpy(), pred.numpy()
    nc = len(classes)
    cm = confusion_matrix(y, p, labels=range(nc))
    pm, rm, fm, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    spec = tn / (tn + fp)
    auc = roc_auc_score(y, probs.numpy(), multi_class="ovr", average="macro", labels=list(range(nc)))

    g = torch.Generator().manual_seed(123)                       # fixed background-swap stress test
    xs = swap_background(x01, Mte, g, p=1.0)
    pr_s = predict(model, xs, mean, std).argmax(1).numpy()

    m = dict(tag=tag, variant=ck["variant"], best_epoch=int(ck["epoch"]), params=count_params(model),
             accuracy=accuracy_score(y, p), precision_macro=pm, recall_macro=rm, f1_macro=fm,
             precision_weighted=pw, recall_weighted=rw, f1_weighted=fw, specificity_macro=float(spec.mean()),
             roc_auc_macro_ovr=auc, ece=ece_score(probs, yte),
             bgswap_accuracy=accuracy_score(y, pr_s), bgswap_f1_macro=f1_score(y, pr_s, average="macro", zero_division=0))

    # CPU latency, batch size 1, single thread
    torch.set_num_threads(1)
    xb = ((x01[:1]) - mean) / std
    with torch.no_grad():
        for _ in range(10): model(xb)
        t = time.time()
        for _ in range(100): model(xb)
    m["latency_ms_cpu_1thread"] = (time.time() - t) / 100 * 1000

    # explanation quality (CNN-style models with a Grad-CAM layer)
    if model.cam_layer is not None:
        rng = np.random.RandomState(0)
        sel = torch.from_numpy(rng.choice(len(yte), min(n_xai, len(yte)), replace=False))
        for name, imgs in [("clean", x01[sel]), ("bgswap", xs[sel])]:
            xn = (imgs - mean) / std
            fr, ratio, dc, dr = [], [], [], []
            for i in range(0, len(sel), 100):
                xb_, mb = xn[i:i + 100], Mte[sel][i:i + 100]
                cam, logits = grad_cam(model, xb_)
                pr = logits.argmax(1)
                a, b = leaf_focus(cam, mb)
                fr.append(a); ratio.append(b)
                dc.append(deletion_auc(model, xb_, cam, pr))
                dr.append(deletion_auc(model, xb_, cam, pr, random_order=True, gen=torch.Generator().manual_seed(1)))
            fr, ratio, dc, dr = map(torch.cat, (fr, ratio, dc, dr))
            m[f"leaf_focus_{name}"] = float(torch.nanmean(fr))
            m[f"leaf_focus_ratio_{name}"] = float(torch.nanmean(ratio))
            m[f"deletion_auc_cam_{name}"] = float(dc.mean())
            m[f"deletion_auc_random_{name}"] = float(dr.mean())
            m[f"deletion_gain_{name}"] = float((dr - dc).mean())
        m["mean_leaf_area_fraction"] = float(Mte[sel].float().mean())

    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    json.dump(m, open(f"results/metrics_{tag}.json", "w"), indent=2)
    pd.DataFrame(classification_report(y, p, labels=range(nc), target_names=classes, output_dict=True,
                                       zero_division=0)).T.assign(
        specificity=list(spec) + [np.nan] * 3).to_csv(f"results/perclass_{tag}.csv")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"results/cm_{tag}.csv")
    plot_cm(cm, classes, f"figures/confusion_{tag}.png", f"Confusion matrix (test) - {tag}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data", default="data/cache_96.npz")
    args = ap.parse_args()
    tags = args.tags or [os.path.basename(f)[:-3] for f in sorted(glob.glob("models/*.pt"))]
    data, classes = load_data(args.data)
    for t in tags:
        m = evaluate_tag(t, data, classes)
        print(t, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k not in ("tag",)}, flush=True)


if __name__ == "__main__":
    main()
