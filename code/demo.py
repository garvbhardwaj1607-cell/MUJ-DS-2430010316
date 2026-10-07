"""Demo: classify one leaf photo and save a Grad-CAM overlay (CPU is fine).

    cd capstone
    python ../code/demo.py --image path/to/leaf.jpg --ckpt models/proposed_mnv3_bgaug.pt --out demo_output.png

The checkpoint is the proposed model: ImageNet-pretrained MobileNetV3-Large fine-tuned on PlantVillage (96x96 input)
with background-randomisation training. It predicts one of 38 crop/disease classes.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image


def grad_cam(model, layer, x):
    acts, grads = {}, {}

    def fwd(_, __, out):
        acts["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    x = x.clone().requires_grad_(True)
    logits = model(x)
    model.zero_grad()
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    h.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return (cam / cam.max().clamp_min(1e-8))[0].detach(), logits.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="models/proposed_mnv3_bgaug.pt")
    ap.add_argument("--out", default="demo_output.png")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["state"])
    model.eval()
    img = Image.open(args.image).convert("RGB").resize((ck["img"], ck["img"]), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()[None] / 255
    xn = (x - ck["mean"]) / ck["std"]
    cam, logits = grad_cam(model, model.blocks[-1], xn)
    prob = logits.softmax(1)[0]
    top = prob.topk(3)
    print(f"Image: {args.image}")
    for p, i in zip(top.values, top.indices):
        print(f"  {classes[int(i)].replace('___', ': ').replace('_', ' '):55s} {float(p):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    ax[0].imshow(img); ax[0].set_title("input (resized)", fontsize=9)
    ax[1].imshow(img); ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title(f"Grad-CAM: {classes[int(top.indices[0])].replace('___', ': ').replace('_', ' ')[:38]}\np={float(top.values[0]):.2f}", fontsize=8)
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(args.out, dpi=130); print("saved", args.out)


if __name__ == "__main__":
    main()
