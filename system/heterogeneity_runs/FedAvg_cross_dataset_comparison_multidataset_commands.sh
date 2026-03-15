#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/run_heterogeneity_comparison.py --dataset-base CINIC10 --dataset-prefix crossds --experiment-name cross_dataset_comparison --goal test --algorithm FedAvg --model MLR --device cpu --device-id 0 --python /home/siguangchen/anaconda3/envs/zyj/bin/python --levels severe --methods fu fedau fedcsa fedosd retrain --partition dir --num-clients 2 --num-classes 10 --global-rounds 1 --local-epochs 1 --batch-size 64 --local-learning-rate 0.005 --join-ratio 1.0 --times 1 --eval-gap 1 --target-client-id 1 --seed 42 --recovery-rounds 5 --max-batches 2 --num-processes 1 --alpha-mild 1.0 --alpha-moderate 0.3 --alpha-severe 0.1 --class-per-client-mild 5 --class-per-client-moderate 3 --class-per-client-severe 2 --balance --continue-on-error --run
