#!/bin/sh
# Fine-tune ImageNet-pretrained models (weights in models/pretrained/, from the official timm GitHub release).
# Resumable (skips finished runs). Two parallel queues, 1 thread each. Run from the project root after preprocessing.
EP=8
job() {
  if [ -f "results/train_$1_s$2.json" ]; then echo "skip $1_s$2"; return; fi
  python ../code/train.py --variant "$1" --seed "$2" --epochs $EP --lr 1e-3 --threads 1
}
q1() { job ref_pre_mnv3 0; job ref_pre_mnv3 1; job ref_pre_mnv3 2; job ref_pre_effb0 0; }
q2() { job pre_mnv3_bgaug 0; job pre_mnv3_bgaug 1; job pre_mnv3_bgaug 2; }
q1 >> logs/queue3.log 2>&1 &
q2 >> logs/queue4.log 2>&1 &
wait
echo PRETRAINED_DONE >> logs/queue3.log
