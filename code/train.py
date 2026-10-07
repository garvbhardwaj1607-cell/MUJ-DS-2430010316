"""Train one variant.  python ../code/train.py --variant ABC_proposed --seed 0 --epochs 20"""
import argparse, os, time, json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import VARIANTS, count_params
from utils import load_data, channel_stats, augment


def evaluate(model, X, y, mean, std, bs=256):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = (X[i:i + bs].float() / 255 - mean) / std
            out = model(x)
            loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
            correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return loss / len(X), correct / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--mask_weight", type=float, default=0.5)
    ap.add_argument("--data", default="data/cache_96.npz")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=0, help="debug: limit batches per epoch")
    ap.add_argument("--suffix", default="", help="appended to the run tag (e.g. _lr1e-3 for sweeps)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    data, classes = load_data(args.data)
    Xtr, Mtr, ytr = data["train"]
    Xva, _, yva = data["val"]
    mean, std = channel_stats(Xtr)

    cfg = VARIANTS[args.variant]
    model = cfg["model"]()
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = int(np.ceil(len(Xtr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    tag = f"{args.variant}_s{args.seed}{args.suffix}"
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    print(f"[{tag}] params={n_params:,} train={len(Xtr)} val={len(Xva)}", flush=True)

    hist, best, t0 = [], (-1, 1e9), time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tl, tc, seen = 0.0, 0, 0
        for b, i in enumerate(range(0, len(Xtr), args.batch)):
            if args.max_batches and b >= args.max_batches:
                break
            idx = perm[i:i + args.batch]
            img, m = augment(Xtr[idx].float() / 255, Mtr[idx], gen, bg_aug=cfg["bg_aug"])
            x = (img - mean) / std
            out = model(x)
            loss = F.cross_entropy(out, ytr[idx])
            if cfg["mask_loss"]:
                target = F.avg_pool2d(m, 8)                       # soft leaf-coverage target at 12x12
                loss = loss + args.mask_weight * F.binary_cross_entropy_with_logits(model.last_att_logits, target)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(idx); tc += (out.argmax(1) == ytr[idx]).sum().item(); seen += len(idx)
        vl, va = evaluate(model, Xva, yva, mean, std)
        hist.append(dict(epoch=ep, train_loss=tl / seen, train_acc=tc / seen, val_loss=vl, val_acc=va,
                         elapsed_s=time.time() - t0))
        print(f"[{tag}] ep {ep:02d} train_loss {tl/seen:.3f} acc {tc/seen:.3f} | val_loss {vl:.3f} acc {va:.3f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        if (va, -vl) > (best[0], -best[1]):
            best = (va, vl)
            torch.save(dict(state=model.state_dict(), variant=args.variant, mean=mean, std=std, classes=classes,
                            epoch=ep), f"models/{tag}.pt")
    pd.DataFrame(hist).to_csv(f"results/history_{tag}.csv", index=False)
    json.dump(dict(tag=tag, params=n_params, best_val_acc=best[0], train_seconds=time.time() - t0,
                   epochs=args.epochs, lr=args.lr, batch=args.batch, wd=args.wd),
              open(f"results/train_{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
