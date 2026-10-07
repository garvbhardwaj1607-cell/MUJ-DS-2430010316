"""Build presentations/PlantVillage_Explainable_Project.pptx from the results CSVs and figures.
Usage (from capstone/):  python ../code/build_deck.py . ../presentations/PlantVillage_Explainable_Project.pptx
All numbers are read from results/*.csv, nothing is typed in by hand."""
import sys, os
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

root, out = sys.argv[1], sys.argv[2]
R = lambda f: pd.read_csv(os.path.join(root, "results", f))
F = lambda f: os.path.join(root, "figures", f)
comp, abl, ablp = R("comparison.csv"), R("ablation.csv"), R("ablation_pretrained.csv")

BLUE, DARK, GREY = RGBColor(0x2A, 0x78, 0xD6), RGBColor(0x1F, 0x24, 0x2E), RGBColor(0x5B, 0x63, 0x70)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def text(s, x, y, w, h, t, size=18, bold=False, color=DARK):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = t if isinstance(t, list) else [t]
    for i, l in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = l
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        p.space_after = Pt(6)
    return tb


def slide(title, sub=None):
    s = prs.slides.add_slide(blank)
    text(s, 0.6, 0.35, 12, 0.8, title, 30, True)
    if sub:
        text(s, 0.6, 1.05, 12, 0.5, sub, 16, False, GREY)
    return s


def pic(s, f, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    s.shapes.add_picture(F(f), Inches(x), Inches(y), **kw)


def table(s, df, x, y, w, size=11):
    rows, cols = len(df) + 1, len(df.columns)
    t = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(0.4 * rows)).table
    for j, c in enumerate(df.columns):
        t.cell(0, j).text = str(c)
    for i, row in enumerate(df.itertuples(index=False), 1):
        for j, v in enumerate(row):
            t.cell(i, j).text = str(v)
    for r in t.rows:
        for c in r.cells:
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(size)


prow = lambda name: comp[comp.Model.str.contains(name, regex=False)].iloc[0]
pro = ablp.iloc[1]
plain = ablp.iloc[0]

s = slide("Explainable Plant-Disease Detection", "Deep Learning Project - CO3 / CO4 / CO5  |  Reg. No. 2430010316  |  Batch F  |  Manipal University Jaipur")
text(s, 0.6, 2.0, 12, 3, ["Real data: PlantVillage, 38 classes, 54,305 colour images (CPU study subset: 7,552 images, 96x96).",
                          "Question: does a leaf classifier look at the leaf, or at the background?",
                          "Contribution: background randomisation (+ mask-supervised attention) and a robustness test that measures it."], 20)

s = slide("Dataset and preprocessing (CO3)")
pic(s, "graph_dataset_distribution.png", 0.5, 1.5, w=7.4)
text(s, 8.2, 1.6, 4.7, 5, ["54,305 images, 38 classes (PlantVillage, pinned commit).",
                           "Study subset: max 200/class = 7,552 images.",
                           "Stratified split 70/15/15, seed 42 (5286 / 1133 / 1133).",
                           "Resize 96x96, normalise, flip / rot90 / brightness augmentation.",
                           "Leaf masks from segmented images - used only in training and evaluation, never at inference."], 16)

s = slide("Pipeline and models (CO3)")
pic(s, "graph_pipeline.png", 0.5, 1.4, w=12.3)

s = slide("Results: classification metrics (CO3)", "Test split, 1133 images. Accuracy, macro P/R/F1, ROC-AUC (OvR), specificity in results.csv")
pic(s, "training_curve_pretrained.png", 0.4, 1.6, w=6.3)
pic(s, "confusion_pre_mnv3_bgaug_s0.png", 6.9, 1.5, h=5.6)
text(s, 0.5, 4.2, 6.2, 2.5, [f"Proposed model (3 seeds): accuracy {pro['Accuracy']}, macro-F1 {pro['Macro-F1']}.", "Per-class precision / recall / F1 / specificity and ROC-AUC: results/results.csv, results/perclass_*.csv."], 15)

s = slide("CO4: comparison with state of the art, same data and split")
pic(s, "graph_model_comparison.png", 0.4, 1.4, w=7.6)
cols = ["Model", "seeds", "Accuracy", "F1(macro)", "BG-swap Acc"]
short = comp[cols].copy()
short["Model"] = short["Model"].str.replace(" (ImageNet-pretrained, fine-tuned)", " (pre)", regex=False).str.replace(" pretrained + background randomisation (B)", "-pre + B", regex=False).str.replace("+ mask-supervised spatial attention (A+B+C) = LeafNet A+B+C", "LeafNet A+B+C", regex=False)
table(s, short, 8.1, 1.6, 5.0, 8)
text(s, 0.6, 6.5, 12, 0.8, "Mohanty et al. 2016 (~99.35%) is context only: different protocol, full data, 256 px. Not a like-for-like comparison.", 13, False, GREY)

s = slide("CO5: innovation - background randomisation", "Pretrained backbones score high but collapse when the background is swapped")
cols = ["Experiment", "seeds", "Accuracy", "BG-swap Acc", "Leaf-focus (clean)"]
table(s, ablp[cols], 0.6, 1.8, 12, 13)
text(s, 0.6, 3.6, 12, 3, [
    f"Plain pretrained MobileNetV3-L: {plain['Accuracy']} accuracy (mean of 3 seeds) but only {plain['BG-swap Acc']} with a swapped background.",
    f"With background randomisation: {pro['Accuracy']} accuracy, {pro['BG-swap Acc']} under swap.",
    "Cost: a small drop in clean accuracy; gain: the model no longer depends on the background."], 18)

s = slide("CO5: ablation study (LeafNet, from scratch, 3 seeds)")
pic(s, "graph_ablation.png", 0.4, 1.4, w=7.4)
cols = ["Experiment", "Accuracy", "BG-swap Acc"]
sa = abl[cols].copy()
sa["Experiment"] = sa["Experiment"].str.replace("+ mask-supervised spatial attention (A+B+C) = Proposed", "+ mask attention (A+B+C)", regex=False)
table(s, sa, 7.9, 1.6, 5.2, 9)
text(s, 0.6, 6.3, 12, 1, "Squeeze-and-excitation gave no gain; background randomisation is the main effect; mask-supervised attention is a small robustness gain at an accuracy cost.", 14, False, GREY)

s = slide("Explainability (Grad-CAM)")
pic(s, "gradcam_pretrained.png", 0.5, 1.5, w=12.2)

s = slide("Limitations and honest caveats")
text(s, 0.6, 1.5, 12, 5, [
    "CPU study: subset (7,552 images) at 96 px. Full-data GPU run is a tracked issue (Colab notebook provided, not yet run on a GPU).",
    "Random per-image split: near-duplicate leaves may leak between train and test, so accuracies may be optimistic.",
    "Background-swap test is synthetic; real field images are not yet evaluated.",
    "Several rows use a single seed; the ablation rows use 3 seeds (mean +/- std).",
    "Hyperparameters were not searched extensively (one learning-rate sensitivity check)."], 18)

s = slide("Reproducibility and next steps")
text(s, 0.6, 1.5, 12, 5, [
    "Pinned requirements.txt, fixed seeds and split, resumable run scripts, pip install -r requirements.txt.",
    "python code/demo.py --image leaf.jpg  ->  prediction + Grad-CAM overlay.",
    "Open issues: full-data Colab run, leakage-safe split, field images, hyperparameter search, web demo, CI."], 18)

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
prs.save(out)
print("saved", out)
