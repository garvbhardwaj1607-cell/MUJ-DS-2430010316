# Deep Learning project: requirement checklist

`assignment_brief.md` is the brief (20 marks: CO3 Apply 10, CO4 Comparative analysis 5, CO5 Innovation 5). This table shows where each requirement is met. Paths are relative to the repository root.

## CO3: implementation (10 marks)
| Requirement | Evidence |
|---|---|
| Dataset selection and preparation | `capstone/README.md` section 3; `code/preprocessing.py`; `capstone/data/dataset_information.txt` |
| Dataset distribution, train/val/test counts and percentages, class imbalance, preprocessing steps | `capstone/results/dataset_distribution.csv`, `split_distribution.csv`, `dataset_info.json`; `capstone/figures/graph_dataset_distribution.png` |
| Model implementation | `code/model.py` |
| Training | `code/train.py`; `capstone/run_all.sh`, `capstone/run_pretrained.sh`; logs in `capstone/logs/` |
| Testing / evaluation | `code/test.py`; `capstone/results/metrics_*.json`, `results.csv` |
| Accuracy, precision, recall, F1 (per class and macro/weighted), specificity, ROC-AUC | `capstone/results/results.csv`, `perclass_*.csv`; table in `capstone/README.md` section 7 |
| Confusion matrix | `capstone/figures/confusion_*.png`, `capstone/results/cm_*.csv` |
| Training vs validation loss and accuracy, learning curves, convergence | `capstone/figures/training_curve*.png`, `learning_curves_ablation.png`; `capstone/results/history_*.csv` |
| Hyperparameter comparison; computational cost; trainable parameters | `capstone/results/lr_sensitivity.csv`; `train_*.json` (seconds); `comparison.csv` (parameters, CPU ms per image) |
| Working source code and reproducible structure | `code/`, `INSTALL.md`, `capstone/requirements.txt` |

## CO4: comparison on the same dataset (5 marks)
| Requirement | Evidence |
|---|---|
| Relevant methods on the same dataset, same split | 8 models in `capstone/results/comparison.csv` (ResNet-style, MobileNet-style, Tiny ViT, LeafNet from scratch; MobileNetV3-L and EfficientNet-B0 pretrained; proposed) |
| Accuracy, precision, recall, F1, AUC, parameters | `capstone/results/comparison.csv`; `capstone/README.md` section 8 |
| Explanation of better / similar / worse | `capstone/README.md` section 8 ("Reading the table") |
| References to the compared papers | `resources/references.md` |
| Full-dataset, 224 px comparison | `notebooks/PlantVillage_pretrained_colab.ipynb` (prepared; run pending, issue 9) |

## CO5: innovation (5 marks)
| Requirement | Evidence |
|---|---|
| Clearly defined, technically justified | `capstone/README.md` section 9 (background randomisation, mask-supervised attention; motivation: PlantVillage background bias) |
| Experimentally evaluated with an ablation table | `capstone/results/ablation.csv` (LeafNet, 3 seeds), `ablation_pretrained.csv` (MobileNetV3-L, 3 seeds); `capstone/figures/graph_ablation.png` |
| Quantitative evidence | background-swap accuracy, leaf-focus (Grad-CAM) and deletion gain in the ablation tables; Grad-CAM figures |
| Not just hyperparameter changes | the contribution changes the training data distribution (B) and adds an attention loss (C) |

## README items required by the brief
Title, student details, dataset, model, hyperparameters, hyperparameter tuning, results, innovation and folder structure are all in `capstone/README.md`. `requirements.txt` has pinned versions.

## Packaging as RegistrationNumber.zip
The brief asks for a ZIP named only with the registration number; the Batch F guidelines ask for a GitHub repository (not only a ZIP). Both can be satisfied: `sh scripts/make_assignment_zip.sh` builds `2430010316.zip` in the layout of the brief (`README.md`, `requirements.txt`, `data/`, `src/`, `notebooks/`, `results/`, `models/`, `figures/`).
