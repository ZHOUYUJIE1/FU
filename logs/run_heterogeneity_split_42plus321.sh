#!/usr/bin/env bash
set -euo pipefail
cd /home/siguangchen/zyj/FUcopy1

/home/siguangchen/anaconda3/envs/zyj/bin/python -u system/run_heterogeneity_comparison.py \
  --run --skip-prepare \
  --dataset-base Cifar10 \
  --dataset-prefix hetero_clean_s42 \
  --experiment-name heterogeneity_clean_seed42_r100_nobd_split \
  --levels mild moderate severe \
  --methods fu fedau fedcsa fedosd retrain \
  --seed 42 \
  --global-rounds 100 \
  --eval-gap 10 \
  --enable-backdoor-attack false \
  --mia-pre-model-source target_local_last \
  --save-target-local-model-last true \
  --continue-on-error

/home/siguangchen/anaconda3/envs/zyj/bin/python -u system/run_heterogeneity_comparison.py \
  --run --skip-prepare \
  --dataset-base Cifar10 \
  --dataset-prefix hetero_clean_s321 \
  --experiment-name heterogeneity_clean_seed321_severe_r100_nobd_split \
  --levels severe \
  --methods fu fedau fedcsa fedosd retrain \
  --seed 321 \
  --global-rounds 100 \
  --eval-gap 10 \
  --enable-backdoor-attack false \
  --mia-pre-model-source target_local_last \
  --save-target-local-model-last true \
  --continue-on-error
