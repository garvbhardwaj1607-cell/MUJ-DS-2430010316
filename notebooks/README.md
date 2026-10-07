# Notebooks

`PlantVillage_pretrained_colab.ipynb` runs the pretrained-model study on the **full** PlantVillage dataset (54,305 images) at 224 px on a Colab GPU, with the same split for all models:
EfficientNet-B0, MobileNetV3-L, ResNet-50, ConvNeXt-Tiny and ViT-Small (CO3/CO4), plus the background-randomisation ablation over 3 seeds (CO5). It writes `comparison.csv`, `ablation.csv`, confusion matrices, curves and Grad-CAM figures.

Run: upload to Colab -> Runtime -> Change runtime type -> GPU -> Run all (roughly 1.5-3 hours on a T4, an estimate).

Status: the code was smoke-tested end to end on CPU with real PlantVillage images (tiny scale, so its accuracies are meaningless). It has **not yet been run on a GPU**; after running it, copy the output tables and figures to `../capstone/results/full_dataset_224/` and commit them (issue 9).
