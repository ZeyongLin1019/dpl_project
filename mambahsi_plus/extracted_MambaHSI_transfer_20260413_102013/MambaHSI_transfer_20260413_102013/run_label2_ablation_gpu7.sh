#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source /home/liushubin/dpl_project/mambahsi_plus/activate311.sh

COMMON_ARGS=(
  --work_dir ./
  --cube_npy_path ./data/label/label1_cube.npy
  --gt_npy_path ./data/label2/label2_gt.npy
  --custom_npy_chunks 64
  --max_epoch 100
  --early_stop_patience 30
  --early_stop_min_delta 0.001
  --use_amp true
  --lr 0.0001
  --train_samples 60
  --val_samples 20
)

# 1) baseline: no class weights, no focal
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python train_MambaHSI_Plus.py \
  --exp_name RUNS_label2_ablate_60x20_baseline_none \
  --class_weight_mode none \
  --use_focal false \
  "${COMMON_ARGS[@]}"

# 2) effective-number weights only
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python train_MambaHSI_Plus.py \
  --exp_name RUNS_label2_ablate_60x20_effective \
  --class_weight_mode effective \
  --class_weight_beta 0.999 \
  --use_focal false \
  "${COMMON_ARGS[@]}"

# 3) effective-number + focal
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python train_MambaHSI_Plus.py \
  --exp_name RUNS_label2_ablate_60x20_effective_focal15 \
  --class_weight_mode effective \
  --class_weight_beta 0.999 \
  --use_focal true \
  --focal_gamma 1.5 \
  "${COMMON_ARGS[@]}"

# 4) effective-number + stronger focal
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python train_MambaHSI_Plus.py \
  --exp_name RUNS_label2_ablate_60x20_effective_focal20 \
  --class_weight_mode effective \
  --class_weight_beta 0.999 \
  --use_focal true \
  --focal_gamma 2.0 \
  "${COMMON_ARGS[@]}"
