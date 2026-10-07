#!/bin/sh
# Reproduce all training experiments (two parallel queues, 1 thread each; written for a 2-core CPU-only machine).
# Resumable: a job is skipped if results/train_<tag>.json already exists.
# Usage (from the project root):
#   python ../code/preprocessing.py --raw <PlantVillage/raw> --out data/cache_96.npz --cap 200 --size 96
#   sh run_all.sh
#   python ../code/test.py --all && python ../code/aggregate.py
EP=12
mkdir -p logs results models

# job <variant> <seed> [extra train.py args...]   (tag suffix for lr sweeps is passed via --suffix)
job() {
  v=$1; s=$2; shift 2
  suffix=""
  case "$*" in *"--suffix "*) suffix=$(echo "$*" | sed 's/.*--suffix \([^ ]*\).*/\1/');; esac
  if [ -f "results/train_${v}_s${s}${suffix}.json" ]; then
    echo "skip ${v}_s${s}${suffix} (already trained)"; return
  fi
  python ../code/train.py --variant "$v" --seed "$s" --epochs $EP --threads 1 "$@"
}

q1() {
  job baseline 0; job AB_se_bgaug 0; job ref_resnet 0
  job baseline 1; job AB_se_bgaug 1
  job baseline 2; job AB_se_bgaug 2
  job ref_tinyvit 0
  # learning-rate sensitivity (baseline, seed 0), reported as a post-hoc check
  job baseline 0 --lr 1e-3 --suffix _lr1e-3
}
q2() {
  job A_se 0; job ABC_proposed 0; job ref_mobilenet 0
  job A_se 1; job ABC_proposed 1
  job A_se 2; job ABC_proposed 2
  job baseline 0 --lr 6e-3 --suffix _lr6e-3
}
q1 >> logs/queue1.log 2>&1 &
q2 >> logs/queue2.log 2>&1 &
wait
echo ALL_TRAINING_DONE >> logs/queue1.log
