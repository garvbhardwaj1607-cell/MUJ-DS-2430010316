# Installation guide

Tested on Linux, Python 3.13, CPU only (2 cores, 7 GB RAM). Python 3.10+ should work.

```bash
git clone https://github.com/<your-username>/MUJ-DS-2430010316.git
cd MUJ-DS-2430010316/capstone
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
GPU users: install the CUDA build of PyTorch first (https://pytorch.org/get-started/locally/), then the rest of `requirements.txt`.

## 1. Get the dataset (real PlantVillage, about 2.8 GB)
```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset ../PlantVillage-Dataset
```
The experiments were run on commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`.

## 2. Pre-process (builds the cached 96x96 subset, leaf masks, split and distribution tables)
```bash
python ../code/preprocessing.py --raw ../PlantVillage-Dataset/raw --out data/cache_96.npz --cap 200 --size 96
```

## 3. Pretrained starting weights (only needed to retrain the pretrained models)
```bash
mkdir -p models/pretrained
curl -L -o models/pretrained/mobilenetv3_large_100.pth https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth
curl -L -o models/pretrained/efficientnet_b0.pth      https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth
```
(The hash prefix in each file name should match the start of the file's SHA-256: `sha256sum models/pretrained/*.pth`.)

## 4. Train, evaluate, aggregate, plot
```bash
sh run_all.sh            # 3 from-scratch reference models + the LeafNet ablation, 3 seeds (about 1.5 h on 2 CPU cores)
sh run_pretrained.sh     # fine-tune pretrained models, 3 seeds for the MobileNetV3 pair (about 1 h on 2 CPU cores)
python ../code/test.py --all          # held-out test-split metrics, confusion matrices, robustness, Grad-CAM metrics
python ../code/aggregate.py           # results/*.csv tables and training-curve / Grad-CAM figures
python ../code/make_graphs.py         # figures/graph_*.png
python ../code/build_readme.py .      # regenerates the README tables from results/*.csv
```
Both scripts are resumable: finished runs are skipped.

## 5. Demo on one image
```bash
python ../code/demo.py --image path/to/leaf.jpg --ckpt models/proposed_mnv3_bgaug.pt --out demo_output.png
```

## 6. Full-dataset run on Google Colab (GPU)
Open `notebooks/PlantVillage_pretrained_colab.ipynb` in Colab, choose a GPU runtime, and run all cells. It trains pretrained models on all 54,305 images at 224 px and writes the same kind of tables and figures. See `notebooks/README.md`.
