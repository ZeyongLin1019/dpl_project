#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python train_MambaHSI_Plus.py \
  --work_dir ./ \
  --exp_name RUNS_cmp_chunks64_lr1e4_weighted_accum \
  --cube_npy_path ./data/label/label1_cube.npy \
  --gt_npy_path ./data/label/label1_gt.npy \
  --custom_npy_chunks 64 \
  --max_epoch 100 \
  --early_stop_patience 30 \
  --early_stop_min_delta 0.001 \
  --use_amp true \
  --lr 0.0001
