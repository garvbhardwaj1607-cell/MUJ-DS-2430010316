"""Shared helpers: data access, augmentation, background randomisation, Grad-CAM, explanation metrics."""
import numpy as np
import torch
import torch.nn.functional as F


def load_data(path="data/cache_96.npz"):
    d = np.load(path, allow_pickle=False)
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).contiguous()      # uint8 N,3,H,W
    M = torch.from_numpy(d["M"]).unsqueeze(1).contiguous()             # uint8 N,1,H,W
    y = torch.from_numpy(d["y"]).long()
    split = d["split"]
    out = {}
    for s in ["train", "val", "test"]:
        i = np.where(split == s)[0]
        out[s] = (X[i], M[i], y[i])
    return out, list(d["classes"])


def channel_stats(X):
    f = X.float() / 255.0
    return f.mean((0, 2, 3)).view(1, 3, 1, 1), f.std((0, 2, 3)).view(1, 3, 1, 1)


def random_background(n, size, gen):
    """Random backgrounds in [0,1]: 1/3 solid colour, 1/3 smooth noise texture, 1/3 two-colour gradient."""
    kind = torch.randint(0, 3, (n,), generator=gen)
    out = torch.empty(n, 3, size, size)
    ramp = torch.linspace(0, 1, size).view(1, 1, size)
    for i in range(n):
        if kind[i] == 0:
            out[i] = torch.rand(3, 1, 1, generator=gen).expand(3, size, size)
        elif kind[i] == 1:
            low = torch.randn(3, 6, 6, generator=gen)
            tex = F.interpolate(low[None], size=size, mode="bilinear", align_corners=False)[0]
            tint = torch.rand(3, 1, 1, generator=gen)
            out[i] = (0.5 + 0.25 * tex) * tint + 0.2 * torch.rand(1, generator=gen)
        else:
            c1, c2 = torch.rand(3, 1, 1, generator=gen), torch.rand(3, 1, 1, generator=gen)
            horiz = torch.rand(1, generator=gen) < 0.5
            t = ramp.expand(1, size, size) if horiz else ramp.transpose(1, 2).expand(1, size, size)
            out[i] = c1 * (1 - t) + c2 * t
    return out.clamp(0, 1)


def swap_background(img01, mask, gen, p=1.0):
    """Replace background (mask==0) by random backgrounds for a fraction p of the samples."""
    bg = random_background(len(img01), img01.shape[-1], gen)
    m = mask.float()
    swapped = img01 * m + bg * (1 - m)
    if p >= 1.0:
        return swapped
    sel = (torch.rand(len(img01), generator=gen) < p).view(-1, 1, 1, 1)
    return torch.where(sel, swapped, img01)


def augment(img01, mask, gen, bg_aug=False, p_bg=0.5):
    """Flip / 90-degree rotation / brightness (all variants); background randomisation (component B)."""
    n = len(img01)
    img, m = img01.clone(), mask.float().clone()
    for i in range(n):
        k = int(torch.randint(0, 4, (1,), generator=gen))
        if k:
            img[i], m[i] = torch.rot90(img[i], k, (1, 2)), torch.rot90(m[i], k, (1, 2))
        if torch.rand(1, generator=gen) < 0.5:
            img[i], m[i] = img[i].flip(2), m[i].flip(2)
    img = (img * (0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=gen))).clamp(0, 1)
    if bg_aug:
        img = swap_background(img, m, gen, p=p_bg)
    return img, m


# ----------------------------------------------------------------------------- Grad-CAM
def grad_cam(model, x, target=None):
    """Grad-CAM at model.cam_layer. Returns (cam [N,H,W] >=0 upsampled to input size, logits)."""
    acts, grads = {}, {}

    def fwd(_, __, out):
        acts["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))

    h = model.cam_layer.register_forward_hook(fwd)
    model.eval()
    x = x.clone().requires_grad_(True)
    logits = model(x)
    tgt = logits.argmax(1) if target is None else target
    model.zero_grad()
    logits.gather(1, tgt[:, None]).sum().backward()
    h.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam.detach(), logits.detach()


def leaf_focus(cam, mask):
    """Fraction of Grad-CAM mass inside the leaf, and that fraction divided by leaf area (1.0 = no preference)."""
    m = mask[:, 0].float()
    tot = cam.flatten(1).sum(1)
    ok = tot > 1e-8
    inside = (cam * m).flatten(1).sum(1)
    frac = torch.where(ok, inside / tot.clamp_min(1e-8), torch.full_like(tot, float("nan")))
    area = m.flatten(1).mean(1).clamp_min(1e-6)
    return frac, frac / area


@torch.no_grad()
def _probs(model, x):
    return F.softmax(model(x), 1)


def deletion_auc(model, x_norm, cam, pred, steps=10, random_order=False, gen=None):
    """Deletion metric: remove the most salient pixels first (replace by dataset mean = 0 after normalisation),
    track the originally predicted class probability. Lower AUC = more faithful explanation."""
    n, _, H, W = x_norm.shape
    if random_order:
        # Like-for-like control: a smooth random saliency map at Grad-CAM's coarse resolution. (Deleting random
        # *individual pixels* creates salt-and-pepper noise that breaks CNNs and makes the comparison unfair.)
        cam = F.interpolate(torch.rand(n, 1, 6, 6, generator=gen), size=(H, W), mode="bilinear",
                            align_corners=False)[:, 0]
    order = cam.flatten(1).argsort(1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(H * W).expand(n, -1))
    curve = []
    for s in range(steps + 1):
        keep = (rank >= int(round(s / steps * H * W))).view(n, 1, H, W).float()
        p = _probs(model, x_norm * keep).gather(1, pred[:, None])[:, 0]
        curve.append(p)
    c = torch.stack(curve, 1)
    return torch.trapezoid(c, dx=1.0 / steps, dim=1)
