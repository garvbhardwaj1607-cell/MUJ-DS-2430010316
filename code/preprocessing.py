"""Dataset preparation for PlantVillage (colour + segmented leaf images).

Builds a cached, resized, stratified subset together with leaf masks derived from the
dataset's own *segmented* images (black background). Masks are used ONLY
 (a) as a training signal for the proposed model (background randomisation, attention supervision)
 (b) to build the background-swap robustness test and the explanation-quality metrics.
The deployed classifier never receives a mask at inference time.

Usage:
    python ../code/preprocessing.py --raw <PlantVillage/raw> --out data/cache_96.npz --cap 200 --size 96
"""
import argparse, os, json
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42


def load_pair(args):
    color_path, seg_path, size = args
    img = Image.open(color_path).convert("RGB")
    seg = np.array(Image.open(seg_path).convert("RGB"))
    mask = (seg.max(axis=2) > 20).astype(np.uint8)
    # dark necrotic lesions can fall below the threshold -> close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    img = np.array(img.resize((size, size), Image.BILINEAR))
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder containing color/ and segmented/")
    ap.add_argument("--out", default="data/cache_96.npz")
    ap.add_argument("--cap", type=int, default=200, help="max images per class (stratified subset)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    rng = np.random.RandomState(SEED)
    classes = sorted(d for d in os.listdir(os.path.join(args.raw, "color")))
    full_counts, items, unpaired = {}, [], {}
    for ci, c in enumerate(classes):
        files = sorted(os.listdir(os.path.join(args.raw, "color", c)))
        full_counts[c] = len(files)
        # Colour and segmented files do not always share the same prefix (some classes, e.g. Corn common rust,
        # have a UUID prefix only on the segmented side) -> pair on the part after the last '___'.
        # Exact filename match first; suffix fallback ONLY when the suffix is unique on both sides.
        seg_dir = os.path.join(args.raw, "segmented", c)
        seg_files = set(os.listdir(seg_dir))
        suffix = lambda name: name.split("___")[-1]
        seg_by_suffix, col_by_suffix = {}, {}
        for sf in seg_files:
            seg_by_suffix.setdefault(suffix(sf[:-len("_final_masked.jpg")]), []).append(sf)
        for f in files:
            col_by_suffix.setdefault(suffix(f.rsplit(".", 1)[0]), []).append(f)
        paired = []
        for f in files:
            exact = f.rsplit(".", 1)[0] + "_final_masked.jpg"
            if exact in seg_files:
                paired.append((f, exact))
                continue
            k = suffix(f.rsplit(".", 1)[0])
            if len(col_by_suffix[k]) == 1 and len(seg_by_suffix.get(k, [])) == 1:
                paired.append((f, seg_by_suffix[k][0]))
        assert len({sf for _, sf in paired}) == len(paired), f"a segmented file was paired twice in {c}"
        unpaired[c] = len(files) - len(paired)
        pick = paired if len(paired) <= args.cap else [paired[j] for j in rng.choice(len(paired), args.cap, replace=False)]
        for f, sf in pick:
            items.append((os.path.join(args.raw, "color", c, f), os.path.join(seg_dir, sf), ci, c, f))
    unpaired = {c: n for c, n in unpaired.items() if n}
    print(f"{len(items)} images selected from {sum(full_counts.values())} total, {len(classes)} classes; "
          f"colour images without a segmented partner (skipped): {unpaired}")
    present = {i[3] for i in items}
    assert present == set(classes), f"classes with no images: {set(classes) - present}"

    with Pool(2) as p:
        res = p.map(load_pair, [(a, b, args.size) for a, b, *_ in items], chunksize=32)
    X = np.stack([r[0] for r in res]).astype(np.uint8)
    M = np.stack([r[1] for r in res]).astype(np.uint8)
    y = np.array([i[2] for i in items])
    names = np.array([i[4] for i in items])

    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=SEED)
    split = np.empty(len(y), dtype="<U5")
    split[tr], split[va], split[te] = "train", "val", "test"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, M=M, y=y, split=split, names=names, classes=np.array(classes))

    os.makedirs(args.results, exist_ok=True)
    df = pd.DataFrame({"class": classes, "full_dataset_images": [full_counts[c] for c in classes]})
    sub = pd.Series(y).map(dict(enumerate(classes))).value_counts()
    df["subset_images"] = df["class"].map(sub).fillna(0).astype(int)
    for s in ["train", "val", "test"]:
        cnt = pd.Series(y[split == s]).map(dict(enumerate(classes))).value_counts()
        df[s] = df["class"].map(cnt).fillna(0).astype(int)
    df["imbalance_ratio_vs_max_full"] = (df["full_dataset_images"].max() / df["full_dataset_images"]).round(2)
    df.to_csv(os.path.join(args.results, "dataset_distribution.csv"), index=False)

    tot = len(y)
    sp = pd.DataFrame({"split": ["train", "val", "test"],
                       "images": [int((split == s).sum()) for s in ["train", "val", "test"]]})
    sp["percent"] = (100 * sp["images"] / tot).round(1)
    sp.to_csv(os.path.join(args.results, "split_distribution.csv"), index=False)

    info = dict(total_full_dataset=int(sum(full_counts.values())), subset=int(tot), classes=len(classes),
                cap_per_class=args.cap, image_size=args.size, mean_leaf_fraction=float(M.mean()),
                colour_images_without_segmented_partner=unpaired,
                seed=SEED, preprocessing=["resize to %dx%d (bilinear)" % (args.size, args.size),
                                          "scale to [0,1], per-channel mean/std normalisation at train time",
                                          "leaf mask from segmented image: max(RGB)>20 then 7x7 morphological closing"])
    json.dump(info, open(os.path.join(args.results, "dataset_info.json"), "w"), indent=2)
    print(sp.to_string(index=False)); print(info)


if __name__ == "__main__":
    main()
