#!/bin/sh
# Creates the project issues with the GitHub CLI. Usage: sh scripts/create_issues.sh <user>/<repo>
set -e
REPO="${1:?usage: sh scripts/create_issues.sh <user>/<repo>}"
for L in "Dataset Collection" "Model Development" "Testing" "Documentation" "Deployment"; do gh label create "$L" --repo "$REPO" 2>/dev/null || true; done

URL=$(gh issue create --repo "$REPO" --title 'Dataset collection and analysis: PlantVillage (colour + segmented)' --label 'Dataset Collection' --body 'Clone PlantVillage, pair colour and segmented images, stratified 70/15/15 split, class-distribution tables and graph.
Evidence: `capstone/results/dataset_distribution.csv`, `split_distribution.csv`, `capstone/figures/graph_dataset_distribution.png`.')
gh issue close "$URL" --repo "$REPO" --comment "Completed in the initial commits; see the commit history."
URL=$(gh issue create --repo "$REPO" --title 'Preprocessing: 96x96 cache and leaf masks' --label 'Dataset Collection' --body '`code/preprocessing.py` builds the cached subset (200 images/class cap) and leaf masks from the segmented images.')
gh issue close "$URL" --repo "$REPO" --comment "Completed in the initial commits; see the commit history."
URL=$(gh issue create --repo "$REPO" --title 'Baseline and reference models trained from scratch (CO3/CO4)' --label 'Model Development' --body 'LeafNet baseline, ResNet-style, MobileNet-style, Tiny ViT, same split and budget. `code/model.py`, `code/train.py`.')
gh issue close "$URL" --repo "$REPO" --comment "Completed in the initial commits; see the commit history."
URL=$(gh issue create --repo "$REPO" --title 'Fine-tune ImageNet-pretrained MobileNetV3-L and EfficientNet-B0 (CO4)' --label 'Model Development' --body 'Pretrained weights from the official timm release; MobileNetV3-L with 3 seeds. `capstone/run_pretrained.sh`.')
gh issue close "$URL" --repo "$REPO" --comment "Completed in the initial commits; see the commit history."
URL=$(gh issue create --repo "$REPO" --title 'Innovation: background randomisation + ablation (CO5)' --label 'Model Development' --body 'Components A (SE), B (background randomisation), C (mask-supervised attention); ablation with 3 seeds on LeafNet and on pretrained MobileNetV3-L.')
gh issue close "$URL" --repo "$REPO" --comment "Completed in the initial commits; see the commit history."
URL=$(gh issue create --repo "$REPO" --title 'Evaluation: metrics, confusion matrices, background-swap test, Grad-CAM leaf focus' --label 'Testing' --body '`code/test.py`, `code/aggregate.py`; outputs in `capstone/results/` and `capstone/figures/`.')
gh issue close "$URL" --repo "$REPO" --comment "Completed in the initial commits; see the commit history."
URL=$(gh issue create --repo "$REPO" --title 'Documentation: README, installation guide, GitHub workflow' --label 'Documentation' --body 'README files generated from result CSVs by `code/build_readme.py`; `INSTALL.md`; `docs/GITHUB_WORKFLOW.md`.')
gh issue close "$URL" --repo "$REPO" --comment "Completed in the initial commits; see the commit history."
URL=$(gh issue create --repo "$REPO" --title 'Demo script: classify one leaf photo with Grad-CAM' --label 'Deployment' --body '`code/demo.py` with the proposed checkpoint `capstone/models/proposed_mnv3_bgaug.pt`.')
gh issue close "$URL" --repo "$REPO" --comment "Completed in the initial commits; see the commit history."
gh issue create --repo "$REPO" --title 'Run the Colab notebook on the full dataset at 224 px and commit the results' --label 'Model Development' --body 'Open `notebooks/PlantVillage_pretrained_colab.ipynb` on a GPU runtime, run all cells, then copy `comparison.csv`, `ablation.csv` and figures into `capstone/results/full_dataset_224/` and update the README (CO3/CO4/CO5 on full data).'
gh issue create --repo "$REPO" --title 'Test for near-duplicate leakage between train and test' --label 'Testing' --body 'PlantVillage has several photos of the same leaf. Check with image hashing or embeddings whether near-duplicates cross the split and, if so, re-split by group and re-report the numbers.'
gh issue create --repo "$REPO" --title 'Evaluate on real field images (for example PlantDoc)' --label 'Testing' --body 'The background-swap test is synthetic. Evaluate the plain and background-randomised models on a real-field dataset restricted to the shared classes.'
gh issue create --repo "$REPO" --title 'Add screenshots of the Colab run and the demo to the documentation' --label 'Documentation' --body 'Add screenshots to `capstone/screenshots/` and reference them from the README.'
gh issue create --repo "$REPO" --title 'Export the presentation to PPTX/PDF and add it to presentations/' --label 'Documentation' --body 'Export the slide deck, commit it, and link it from the README.'
gh issue create --repo "$REPO" --title 'Hyperparameter search for the pretrained models' --label 'Model Development' --body 'Learning rate, epochs, resolution (96 vs 160 vs 224), label smoothing; select on validation only; report in the README tuning section.'
gh issue create --repo "$REPO" --title 'Web demo (Gradio or Streamlit) around code/demo.py' --label 'Deployment' --body 'Upload a leaf photo, show top-3 classes and the Grad-CAM overlay; deploy to Hugging Face Spaces or Streamlit Cloud.'
gh issue create --repo "$REPO" --title 'Continuous integration: lint and a smoke test' --label 'Testing' --body 'GitHub Actions workflow that installs requirements and runs a tiny training smoke test.'
