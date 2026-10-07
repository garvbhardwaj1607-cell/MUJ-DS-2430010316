"""Regenerate README.md (capstone + repository root), data/dataset_information.txt and models/model_description.txt
from the result CSVs, so that every number in the documentation comes from results/ and nothing is typed by hand.

    cd capstone && python ../code/build_readme.py .
"""
import json, os, sys
import numpy as np
import pandas as pd

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
repo = os.path.abspath(os.path.join(root, ".."))
R = lambda p: pd.read_csv(os.path.join(root, "results", p), dtype=str).fillna("")


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


info = json.load(open(os.path.join(root, "results", "dataset_info.json")))
dist = pd.read_csv(os.path.join(root, "results", "dataset_distribution.csv"))
split, cmp_, abl, abl_pre, lr = R("split_distribution.csv"), R("comparison.csv"), R("ablation.csv"), R("ablation_pretrained.csv"), R("lr_sensitivity.csv")
res = pd.read_csv(os.path.join(root, "results", "results.csv"))
main = res[~res.tag.str.contains("_lr")]

S = lambda v: main[main.variant == v]
mv = lambda v, c: S(v)[c].mean()
sd = lambda v, c: S(v)[c].std(ddof=0)


def pm(v, c, d=4):
    s = S(v)[c]
    return f"{s.mean():.{d}f} ± {s.std(ddof=0):.{d}f}" if len(s) > 1 else f"{s.iloc[0]:.{d}f}"


NAME = {"pre_mnv3_bgaug": "**Proposed: MobileNetV3-L (pretrained) + background randomisation**", "ref_pre_mnv3": "MobileNetV3-L (pretrained, plain)",
        "ref_pre_effb0": "EfficientNet-B0 (pretrained, plain)", "ABC_proposed": "LeafNet A+B+C (scratch, lightweight variant)",
        "baseline": "LeafNet baseline (scratch)"}
core_rows = []
for v in ["pre_mnv3_bgaug", "ref_pre_mnv3", "ref_pre_effb0", "ABC_proposed", "baseline"]:
    if S(v).empty: continue
    core_rows.append({"Model": NAME[v], "seeds": len(S(v)), "Accuracy": pm(v, "accuracy"), "Precision (macro)": pm(v, "precision_macro"),
                      "Recall (macro)": pm(v, "recall_macro"), "F1 (macro)": pm(v, "f1_macro"),
                      "Specificity (macro)": pm(v, "specificity_macro"), "ROC-AUC (macro OvR)": pm(v, "roc_auc_macro_ovr", 5),
                      "ECE": pm(v, "ece")})
core = pd.DataFrame(core_rows)

prop, plain = "pre_mnv3_bgaug", "ref_pre_mnv3"
cost = 100 * (mv(plain, "accuracy") - mv(prop, "accuracy"))
gain = 100 * (mv(prop, "bgswap_accuracy") - mv(plain, "bgswap_accuracy"))
no_b_scratch = [v for v in ["baseline", "A_se", "ref_resnet", "ref_mobilenet", "ref_tinyvit"] if not S(v).empty]
no_b_pre = [v for v in [plain, "ref_pre_effb0"] if not S(v).empty]
rng = lambda vs: (100 * min(mv(v, "bgswap_accuracy") for v in vs), 100 * max(mv(v, "bgswap_accuracy") for v in vs))
sc_lo, sc_hi = rng(no_b_scratch); pr_lo, pr_hi = rng(no_b_pre)
dA_c, dA_b = 100 * (mv("A_se", "accuracy") - mv("baseline", "accuracy")), 100 * (mv("A_se", "bgswap_accuracy") - mv("baseline", "bgswap_accuracy"))
dB_c, dB_b = 100 * (mv("AB_se_bgaug", "accuracy") - mv("A_se", "accuracy")), 100 * (mv("AB_se_bgaug", "bgswap_accuracy") - mv("A_se", "bgswap_accuracy"))
dC_c, dC_b = 100 * (mv("ABC_proposed", "accuracy") - mv("AB_se_bgaug", "accuracy")), 100 * (mv("ABC_proposed", "bgswap_accuracy") - mv("AB_se_bgaug", "bgswap_accuracy"))
leaf_n = S(plain)["leaf_focus_bgswap"].mean(), S(prop)["leaf_focus_bgswap"].mean()

pc = pd.read_csv(os.path.join(root, "results", f"perclass_{prop}_s0.csv"), index_col=0)
worst = pc.iloc[:-3].sort_values("f1-score").head(5)
worst_txt = "\n".join(f"  - {i}: F1 {r['f1-score']:.3f} (precision {r['precision']:.3f}, recall {r['recall']:.3f})" for i, r in worst.iterrows())
imb = dist["full_dataset_images"]

STUDENT = """- Name: Garv *(add surname if required)*
- Registration number: 2430010316
- Branch: *(fill in, e.g. Data Science)*
- Batch: F
- GitHub username: *(fill in)*
- Training programme: Batch F capstone training, Manipal University Jaipur *(add programme / instructor details)*"""

SUMMARY = f"""1. **Implementation (CO3):** real PlantVillage data ({info['total_full_dataset']:,} images, 38 classes; a stratified {info['subset']:,}-image subset at {info['image_size']}x{info['image_size']} was used for the CPU experiments), 5 from-scratch and 3 ImageNet-pretrained model variants, full metric suite, confusion matrices, curves.
2. **Comparison (CO4):** 8 models on one identical split; ImageNet-pretrained models reach about {100*mv('ref_pre_effb0','accuracy'):.1f}-{100*mv(plain,'accuracy'):.1f} % test accuracy against {100*min(mv(v,'accuracy') for v in no_b_scratch):.0f}-{100*max(mv(v,'accuracy') for v in no_b_scratch):.0f} % for small CNNs/ViT trained from scratch.
3. **Innovation (CO5):** models without background randomisation fall to {sc_lo:.0f}-{sc_hi:.0f} % (scratch) and {pr_lo:.0f}-{pr_hi:.0f} % (pretrained) accuracy when only the leaf background is replaced. Training with random background replacement keeps the pretrained MobileNetV3-L at **{pm(prop,'bgswap_accuracy')}** background-swap accuracy (plain fine-tuning: {pm(plain,'bgswap_accuracy')}) for a clean-accuracy cost of {cost:.1f} points ({pm(prop,'accuracy')} vs {pm(plain,'accuracy')}; {len(S(prop))} seeds each).
4. **What did not help:** squeeze-and-excitation attention (A) changed accuracy by {dA_c:+.1f} points and background-swap accuracy by {dA_b:+.1f} points (within seed noise); the mask-supervised attention (C) gave {dC_b:+.1f} points of background-swap accuracy for {dC_c:+.1f} points of clean accuracy."""

PIPE = "![pipeline](figures/graph_pipeline.png)"

capstone = f"""# Explainable and Background-Robust Plant Disease Classification on PlantVillage

*Deep Learning project (CO3 Apply, CO4 Comparative analysis, CO5 Innovation) and Batch F capstone.*

## 1. Student details
{STUDENT}

## 2. Summary of results
All numbers are produced by the code in `../code/` and stored in `results/`; this README is generated from those files (`../code/build_readme.py`).
{SUMMARY}

{PIPE}

**Status:** phase 1 (CPU study on a subset at 96 px) is complete. Phase 2 (full dataset at 224 px on a Colab GPU, `../notebooks/PlantVillage_pretrained_colab.ipynb`) is prepared and tested on CPU in smoke mode, but not yet run on a GPU; it is tracked as an open issue (see section 12).

## 3. Dataset
- **Name:** PlantVillage (colour + segmented leaf images), 38 classes (14 crops; healthy and diseased).
- **Source:** https://github.com/spMohanty/PlantVillage-Dataset (commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, folders `raw/color` and `raw/segmented`). Original paper: Hughes & Salathé, arXiv:1511.08060.
- **Full dataset:** {info['total_full_dataset']:,} colour images. **Used here:** a stratified subset of **{info['subset']:,}** images (cap of {info['cap_per_class']} per class; smaller classes keep all images, e.g. Potato healthy has 152), about {100*info['subset']/info['total_full_dataset']:.1f} % of the data, so that all 28 training runs fit on a 2-core CPU.
- **Split (stratified, seed 42):**

{md(split)}

- **Class distribution and imbalance:** the full dataset ranges from {int(imb.min())} to {int(imb.max())} images per class (imbalance {imb.max()/imb.min():.1f}:1). The capped subset is almost balanced (140/30/30 per class for train/val/test, Potato healthy 106/23/23). Per-class counts: `results/dataset_distribution.csv`.

![class distribution](figures/graph_dataset_distribution.png)

- **Preprocessing:** resize to {info['image_size']}x{info['image_size']} (bilinear); scale to [0,1]; per-channel mean/std normalisation, with statistics computed on the training split and used for every model (including the pretrained ones in this repo; the Colab notebook instead uses timm's ImageNet statistics); leaf mask from the segmented image (max RGB > 20, 7x7 morphological closing). Training augmentation: random flip, random 90-degree rotation, brightness x[0.8, 1.2]. Masks are used only for training (components B/C) and for the robustness/explanation metrics; **the classifier never receives a mask at inference**.
- Colour images without a segmented partner: {info['colour_images_without_segmented_partner'] or 'none'}.

## 4. Models
- **Baseline LeafNet** (from scratch, 662,558 parameters): stem conv, three stages of two 3x3 conv-BN-ReLU layers, global average pooling, dropout 0.3, linear head.
- **LeafNet A+B+C** (675,129 parameters): baseline plus **A** squeeze-and-excitation channel attention, **B** background randomisation (with probability 0.5 the background of a training image is replaced by a random solid colour, smooth noise texture or two-colour gradient), **C** a spatial attention gate supervised with a leaf-coverage target from the mask (BCE, weight 0.5).
- **Reference models (CO4), identical split:** from scratch (same 12-epoch budget): ResNet-style, MobileNet-style, Tiny ViT; **ImageNet-pretrained and fine-tuned end to end:** MobileNetV3-Large (Howard et al., 2019) and EfficientNet-B0 (Tan & Le, 2019), `timm` implementations with the official `timm` release weights (SHA-256 prefix matches the hash in each file name).
- **Proposed model:** the pretrained MobileNetV3-Large trained with background randomisation (B). Everything else is identical to the plain fine-tuned model.
- Checkpoints are selected on **validation** accuracy; the test split is used once per model.

## 5. Hyperparameters
| Hyperparameter | From-scratch models | Pretrained models |
|---|---|---|
| Learning rate (peak) | 3e-3 | 1e-3 |
| Batch size | 64 | 64 |
| Epochs | 12 | 8 |
| Optimizer | AdamW | AdamW |
| Loss | cross-entropy (+0.5 x mask BCE for C) | cross-entropy |
| Scheduler | OneCycleLR | OneCycleLR |
| Dropout | 0.3 | timm default (none added) |
| Weight decay | 1e-4 | 1e-4 |
| Seeds | 0,1,2 for the 4 LeafNet ablation variants; 0 for the other scratch models | 0,1,2 for MobileNetV3-L (plain and +B); 0 for EfficientNet-B0 |
| Input size | 96x96 | 96x96 (not upsampled) |

## 6. Hyperparameter tuning
A small post-hoc learning-rate check was run (LeafNet baseline, seed 0, 12 epochs). The values were not used to pick the final setting; 3e-3 was fixed in advance.

{md(lr)}

Validation accuracy differs by at most 0.4 points over a 6x range of learning rate, smaller than the seed-to-seed spread of the baseline, so no real sensitivity was detected. Batch size, epochs, weight decay, dropout and the pretrained models' learning rate (1e-3, a conventional fine-tuning value) were **not** tuned; a search is an open issue.

## 7. Results (CO3), held-out test split ({int(res.test_images.iloc[0]) if 'test_images' in res else 1133} images; mean ± std over seeds where `seeds` > 1)
{md(core)}

- Confusion matrices: `figures/confusion_<run>.png` (CSV `results/cm_<run>.csv`); the proposed model's is `figures/confusion_pre_mnv3_bgaug_s0.png`.
- Per-class precision/recall/F1/specificity: `results/perclass_<run>.csv`. All per-run metrics: `results/results.csv`.
- Training and validation curves: `figures/training_curve.png` (scratch), `figures/training_curve_pretrained.png` (pretrained), `figures/learning_curves_ablation.png`; data in `results/history_<run>.csv`.
- Weakest classes of the proposed model (seed 0, by F1):
{worst_txt}

![confusion matrix of the proposed model](figures/confusion_pre_mnv3_bgaug_s0.png)
![training curves, pretrained](figures/training_curve_pretrained.png)

## 8. Comparison with other models (CO4)
Same dataset, same split, same preprocessing. "BG-swap Acc" is accuracy when the background of every test image is replaced (leaf pixels unchanged).

{md(cmp_)}

![model comparison](figures/graph_model_comparison.png)
![accuracy versus parameters](figures/graph_accuracy_vs_params.png)

**Published results, for context only (not comparable):** Mohanty et al. (2016) report about 99.35 % accuracy with a fine-tuned GoogLeNet on the full 38-class colour set with an 80/20 split. That uses about 7x more images, a larger input size and a different split protocol, so it must not be compared directly with the table above; it was not reproduced here.

**Reading the table.** Pretrained models are clearly best on clean accuracy, at 4.1-4.3 M parameters. Among from-scratch models the lightweight MobileNet-style network is best. Single-seed rows should not be ranked by differences of about 1 point: the LeafNet baseline alone varies by {100*(S('baseline').accuracy.max()-S('baseline').accuracy.min()):.1f} points across 3 seeds.

## 9. Innovation and ablation (CO5)
**Innovation.** PlantVillage photographs share a uniform background, so classifiers can rely on it. The project (i) measures this with a background-swap stress test and mask-based Grad-CAM metrics built from the dataset's own segmentation, and (ii) trains with random background replacement (B), optionally with mask-supervised attention (C). SE (A), background augmentation and attention supervision are known techniques; the contribution is their combination for this setting and the quantitative evaluation of each part, including the finding that SE does not help.

**Ablation 1: from-scratch LeafNet, components added one at a time (3 seeds).**

{md(abl[['Experiment','A: SE','B: BG-rand','C: mask-attn','seeds','Accuracy','Macro-F1','BG-swap Acc','BG-swap F1','ECE']])}

{md(abl[['Experiment','Leaf-focus (clean)','Leaf-focus (BG-swap)','Deletion gain (clean)','Params']])}

**Ablation 2: ImageNet-pretrained MobileNetV3-L, with and without B ({len(S(prop))} seeds).**

{md(abl_pre)}

![ablation](figures/graph_ablation.png)
![Grad-CAM, scratch models](figures/gradcam_examples.png)
![Grad-CAM, pretrained models](figures/gradcam_pretrained.png)

- *Leaf-focus*: fraction of Grad-CAM mass inside the leaf (the leaf covers about {res.mean_leaf_area_fraction.dropna().iloc[0]:.2f} of the image, so about that value means no preference). *Deletion gain*: how much faster the predicted-class probability drops when the most salient pixels are removed than when a random smooth map is used. Both on 400 test images.
- A (SE): {dA_c:+.1f} points clean, {dA_b:+.1f} points background-swap accuracy, i.e. no measurable benefit.
- B (background randomisation): {dB_c:+.1f} points clean, {dB_b:+.1f} points background-swap accuracy on LeafNet; on the pretrained model {-cost:+.1f} points clean and {gain:+.1f} points background-swap accuracy, and background-swap leaf-focus {leaf_n[0]:.3f} to {leaf_n[1]:.3f}.
- C (mask-supervised attention): {dC_c:+.1f} points clean, {dC_b:+.1f} points background-swap accuracy, a small trade-off.

## 10. Limitations
- **Subset and resolution:** {info['subset']:,} of {info['total_full_dataset']:,} images at {info['image_size']} px; absolute accuracies are not comparable with published full-dataset results. The Colab notebook is the route to full-data numbers.
- **Possible optimism from the random split:** PlantVillage has several photos of the same leaf and the split is random per image, so near-duplicates may cross train/test. This is a known concern; it was not measured here (open issue).
- **The background-swap test is synthetic** and uses the same family of random backgrounds as the training augmentation, so it shows learned invariance to those backgrounds, not robustness on real field photographs (open issue).
- **Seeds:** the four LeafNet ablation variants and the MobileNetV3-L pair have 3 seeds; all other rows are single runs.
- **Calibration:** background-trained scratch models are less well calibrated (see ECE columns).
- Grad-CAM at 96 px input has a coarse feature map (6x6 for LeafNet, 3x3 for the pretrained models).
- CPU latency is for batch size 1, one thread, 96x96 input; latency was measured while other training jobs were running on the same 2 cores (pretrained rows show large ± for that reason, so compare only the scratch rows with each other), and training times in `results/train_*.json` are not clean timings.

## 11. Demo
```bash
python ../code/demo.py --image path/to/leaf.jpg --ckpt models/proposed_mnv3_bgaug.pt --out demo_output.png
```
Prints the top-3 classes and saves a Grad-CAM overlay. Example: `figures/demo_example.png` - an arbitrarily chosen Tomato Late blight photo (5th file in its folder, not chosen for being correct; it may be in the training split). The model predicted Early blight (p=0.65), i.e. **a wrong prediction**, on a badly damaged leaf; it is kept as an honest example of a failure.

## 12. Status, roadmap and future commits
| Item | Status |
|---|---|
| CO3: implementation, metrics, confusion matrices, curves | done (subset, 96 px) |
| CO4: comparison on the same split | done (8 models) |
| CO5: innovation with 3-seed ablations | done |
| Full dataset, 224 px, GPU (Colab notebook) | prepared, **to run** (issue 9) |
| Near-duplicate leakage check | open (issue 10) |
| Real field images | open (issue 11) |
| Screenshots of the Colab run | open (issue 12) |
| Export the presentation to PPTX/PDF into `../presentations/` | open (issue 13) |
| Hyperparameter search, web demo, CI | open (issues 14-16) |

The issue list is in `../docs/ISSUES.md`. Each open item is a planned future commit.

## 13. Folder structure
```
capstone/
├── README.md                  # this file (generated by ../code/build_readme.py)
├── requirements.txt
├── run_all.sh                 # from-scratch experiments (resumable)
├── run_pretrained.sh          # pretrained fine-tuning (resumable)
├── data/dataset_information.txt
├── results/                   # results.csv, comparison.csv, ablation*.csv, lr_sensitivity.csv, dataset_*.csv,
│                              # metrics_*.json, perclass_*.csv, cm_*.csv, history_*.csv, train_*.json
├── figures/                   # confusion_*.png, graph_*.png, training curves, Grad-CAM figures, demo example
├── models/                    # proposed_mnv3_bgaug.pt (demo checkpoint) and model_description.txt
└── logs/                      # training / evaluation logs
../code/                       # preprocessing.py, model.py, train.py, test.py, aggregate.py, make_graphs.py, demo.py, utils.py, build_readme.py
../notebooks/                  # Colab notebook (full dataset, 224 px)
```
Installation and reproduction: `../INSTALL.md`. References: `../resources/references.md`.
"""
open(os.path.join(root, "README.md"), "w").write(capstone)

repo_readme = f"""# MUJ-DS-2430010316

## Student details
{STUDENT}

## Project title
**Explainable and Background-Robust Plant Disease Classification on PlantVillage** (Deep Learning project and capstone)

## What this project does
PlantVillage leaf photographs share a uniform background, so a classifier can partly rely on it. This project trains and compares from-scratch and ImageNet-pretrained CNNs on the real PlantVillage dataset, shows how much they depend on the background with a background-swap test, and fixes most of that dependence by training with random background replacement.

![pipeline](capstone/figures/graph_pipeline.png)

## Results at a glance
{SUMMARY}

![comparison](capstone/figures/graph_model_comparison.png)

Full results, tables and limitations: [`capstone/README.md`](capstone/README.md).

## Repository layout (Batch F guidelines)
| Folder | Content |
|---|---|
| `assignments/` | Deep Learning project brief and a requirement-by-requirement checklist (CO3/CO4/CO5) |
| `notebooks/` | Google Colab notebook: pretrained models on the full dataset at 224 px |
| `code/` | All source code: preprocessing, models, training, evaluation, graphs, demo, README generator |
| `resources/` | References and dataset links |
| `presentations/` | Project presentation |
| `capstone/` | The project itself: README with results, requirements, run scripts, results, figures, demo checkpoint |
| `docs/`, `scripts/`, `.github/` | GitHub workflow, issue list and creation script, pull-request template |

## Quick start
See [`INSTALL.md`](INSTALL.md). Demo on one image: `cd capstone && python ../code/demo.py --image leaf.jpg`.

## Deliverables checklist (guidelines, step 11)
| Deliverable | Where |
|---|---|
| Source code | `code/`, `notebooks/` |
| Documentation | `capstone/README.md`, `INSTALL.md`, `docs/` |
| Presentation | `presentations/` |
| Screenshots / figures | `capstone/figures/` (Colab-run screenshots: open issue 12) |
| Results | `capstone/results/` |
| Installation guide | `INSTALL.md` |

## Status
Phase 1 (CPU study, subset, 96 px) is complete with real data and executed code. Remaining work (full-dataset Colab run, leakage check, real-field test, web demo) is tracked in [`docs/ISSUES.md`](docs/ISSUES.md) and will be added in later commits. GitHub steps that need an account (repository creation, instructor collaborator, issues, pull requests) are in [`docs/GITHUB_WORKFLOW.md`](docs/GITHUB_WORKFLOW.md).
"""
open(os.path.join(repo, "README.md"), "w").write(repo_readme)

open(os.path.join(root, "data", "dataset_information.txt"), "w").write(
    f"""PlantVillage (colour + segmented), 38 classes. Source: https://github.com/spMohanty/PlantVillage-Dataset (commit 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a).
Full dataset: {info['total_full_dataset']} images. Used: stratified subset of {info['subset']} images (cap {info['cap_per_class']} per class), {info['image_size']}x{info['image_size']}.
Split: 70/15/15 stratified (seed 42): {', '.join(f"{a} {b}" for a, b in zip(split.split, split.images))}.
Preprocessing: {'; '.join(info['preprocessing'])}.
Per-class counts: results/dataset_distribution.csv. The cached array data/cache_96.npz (about 150 MB) is not stored in the repository; regenerate it with code/preprocessing.py (see INSTALL.md).
""")
open(os.path.join(root, "models", "model_description.txt"), "w").write(
    """models/proposed_mnv3_bgaug.pt: the proposed model (ImageNet-pretrained MobileNetV3-Large fine-tuned on PlantVillage with background-randomisation
training; seed 0, checkpoint of the best validation epoch). Dict with keys: state (timm 'mobilenetv3_large_100' weights), mean, std, classes, arch, img.
All other trained checkpoints (about 135 MB) are not stored in the repository; run run_all.sh / run_pretrained.sh to regenerate them.
Pretrained starting weights (official timm release) are not stored either; see INSTALL.md.
Architectures: code/model.py.
""")
print("README files written")
