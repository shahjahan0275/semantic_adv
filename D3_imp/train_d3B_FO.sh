#!/bin/bash
# Job script for running 3-branch FOS training on h100-dell
# Usage: nohup bash train_d3B_FO.sh &

set -e  # Exit immediately if a command fails

# ----------------------------
# Set paths
# ----------------------------
PROJECT_DIR=/forcolab/home/ashahj/D3
LOG_DIR=$PROJECT_DIR/logs
CKPT_DIR=$PROJECT_DIR/ckpt
PYTHON_SCRIPT=$PROJECT_DIR/train_3branch_FOS.py
LOG_FILE=$LOG_DIR/train_d3B_FO_$(date +%Y%m%d_%H%M%S).log

# ----------------------------
# Activate conda environment
# ----------------------------
source /forcolab/home/ashahj/miniconda3/etc/profile.d/conda.sh
conda activate D3

# ----------------------------
# Make sure directories exist
# ----------------------------
mkdir -p $LOG_DIR
mkdir -p $CKPT_DIR

# ----------------------------
# Run training
# ----------------------------
echo "Starting training at $(date)" | tee -a $LOG_FILE

python $PYTHON_SCRIPT \
    --name=train_d3B_FO \
    --train_samples=18000 \
    --arch=CLIP:ViT-L/14 \
    --checkpoints_dir=$CKPT_DIR \
    --fix_backbone \
    --head_type=attention \
    --batch_size=64 \
    --shuffle \
    --patch_size=14 \
    >> $LOG_FILE 2>&1

echo "Training finished at $(date)" | tee -a $LOG_FILE


