# Issues

Create them on GitHub with `sh scripts/create_issues.sh <user>/MUJ-DS-2430010316` (needs the GitHub CLI, `gh auth login`).
Items marked done are created and closed immediately so the issue history shows the work already completed.

| # | Title | Label | Status |
|---|---|---|---|
| 1 | Dataset collection and analysis: PlantVillage (colour + segmented) | Dataset Collection | done |
| 2 | Preprocessing: 96x96 cache and leaf masks | Dataset Collection | done |
| 3 | Baseline and reference models trained from scratch (CO3/CO4) | Model Development | done |
| 4 | Fine-tune ImageNet-pretrained MobileNetV3-L and EfficientNet-B0 (CO4) | Model Development | done |
| 5 | Innovation: background randomisation + ablation (CO5) | Model Development | done |
| 6 | Evaluation: metrics, confusion matrices, background-swap test, Grad-CAM leaf focus | Testing | done |
| 7 | Documentation: README, installation guide, GitHub workflow | Documentation | done |
| 8 | Demo script: classify one leaf photo with Grad-CAM | Deployment | done |
| 9 | Run the Colab notebook on the full dataset at 224 px and commit the results | Model Development | open |
| 10 | Test for near-duplicate leakage between train and test | Testing | open |
| 11 | Evaluate on real field images (for example PlantDoc) | Testing | open |
| 12 | Add screenshots of the Colab run and the demo to the documentation | Documentation | open |
| 13 | Export the presentation to PPTX/PDF and add it to presentations/ | Documentation | open |
| 14 | Hyperparameter search for the pretrained models | Model Development | open |
| 15 | Web demo (Gradio or Streamlit) around code/demo.py | Deployment | open |
| 16 | Continuous integration: lint and a smoke test | Testing | open |

## Details

### 1. Dataset collection and analysis: PlantVillage (colour + segmented)
*Label:* Dataset Collection  |  *Status:* done

Clone PlantVillage, pair colour and segmented images, stratified 70/15/15 split, class-distribution tables and graph.
Evidence: `capstone/results/dataset_distribution.csv`, `split_distribution.csv`, `capstone/figures/graph_dataset_distribution.png`.

### 2. Preprocessing: 96x96 cache and leaf masks
*Label:* Dataset Collection  |  *Status:* done

`code/preprocessing.py` builds the cached subset (200 images/class cap) and leaf masks from the segmented images.

### 3. Baseline and reference models trained from scratch (CO3/CO4)
*Label:* Model Development  |  *Status:* done

LeafNet baseline, ResNet-style, MobileNet-style, Tiny ViT, same split and budget. `code/model.py`, `code/train.py`.

### 4. Fine-tune ImageNet-pretrained MobileNetV3-L and EfficientNet-B0 (CO4)
*Label:* Model Development  |  *Status:* done

Pretrained weights from the official timm release; MobileNetV3-L with 3 seeds. `capstone/run_pretrained.sh`.

### 5. Innovation: background randomisation + ablation (CO5)
*Label:* Model Development  |  *Status:* done

Components A (SE), B (background randomisation), C (mask-supervised attention); ablation with 3 seeds on LeafNet and on pretrained MobileNetV3-L.

### 6. Evaluation: metrics, confusion matrices, background-swap test, Grad-CAM leaf focus
*Label:* Testing  |  *Status:* done

`code/test.py`, `code/aggregate.py`; outputs in `capstone/results/` and `capstone/figures/`.

### 7. Documentation: README, installation guide, GitHub workflow
*Label:* Documentation  |  *Status:* done

README files generated from result CSVs by `code/build_readme.py`; `INSTALL.md`; `docs/GITHUB_WORKFLOW.md`.

### 8. Demo script: classify one leaf photo with Grad-CAM
*Label:* Deployment  |  *Status:* done

`code/demo.py` with the proposed checkpoint `capstone/models/proposed_mnv3_bgaug.pt`.

### 9. Run the Colab notebook on the full dataset at 224 px and commit the results
*Label:* Model Development  |  *Status:* open

Open `notebooks/PlantVillage_pretrained_colab.ipynb` on a GPU runtime, run all cells, then copy `comparison.csv`, `ablation.csv` and figures into `capstone/results/full_dataset_224/` and update the README (CO3/CO4/CO5 on full data).

### 10. Test for near-duplicate leakage between train and test
*Label:* Testing  |  *Status:* open

PlantVillage has several photos of the same leaf. Check with image hashing or embeddings whether near-duplicates cross the split and, if so, re-split by group and re-report the numbers.

### 11. Evaluate on real field images (for example PlantDoc)
*Label:* Testing  |  *Status:* open

The background-swap test is synthetic. Evaluate the plain and background-randomised models on a real-field dataset restricted to the shared classes.

### 12. Add screenshots of the Colab run and the demo to the documentation
*Label:* Documentation  |  *Status:* open

Add screenshots to `capstone/screenshots/` and reference them from the README.

### 13. Export the presentation to PPTX/PDF and add it to presentations/
*Label:* Documentation  |  *Status:* open

Export the slide deck, commit it, and link it from the README.

### 14. Hyperparameter search for the pretrained models
*Label:* Model Development  |  *Status:* open

Learning rate, epochs, resolution (96 vs 160 vs 224), label smoothing; select on validation only; report in the README tuning section.

### 15. Web demo (Gradio or Streamlit) around code/demo.py
*Label:* Deployment  |  *Status:* open

Upload a leaf photo, show top-3 classes and the Grad-CAM overlay; deploy to Hugging Face Spaces or Streamlit Cloud.

### 16. Continuous integration: lint and a smoke test
*Label:* Testing  |  *Status:* open

GitHub Actions workflow that installs requirements and runs a tiny training smoke test.
