#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source /home/liushubin/dpl_project/mambahsi_plus/activate311.sh

CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python train_MambaHSI_Plus.py \
  --work_dir ./ \
  --exp_name RUNS_label2_60x20_effective_focal15 \
  --cube_npy_path ./data/label/label1_cube.npy \
  --gt_npy_path ./data/label2/label2_gt.npy \
  --custom_npy_chunks 64 \
  --max_epoch 100 \
  --early_stop_patience 30 \
  --early_stop_min_delta 0.001 \
  --use_amp true \
  --lr 0.0001 \
  --train_samples 60 \
  --val_samples 20 \
  --class_weight_mode effective \
  --class_weight_beta 0.999 \
  --use_focal true \
  --focal_gamma 1.5
