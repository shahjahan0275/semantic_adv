#!/bin/bash
#SBATCH --job-name=Resynthesis_styleclip_DCT        # Job name
#SBATCH --mail-type=ALL                             # Email notifications
#SBATCH --chdir=/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis  # Working directory
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=16                          # CPU cores per task
#SBATCH --mem=128G                                  # Memory per node
#SBATCH --time=168:00:00                            # Maximum runtime (168 hours)
#SBATCH --account=mmannan                            # Account name
#SBATCH --partition=pt                               # GPU partition
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1           # Request 1 GPU

#SBATCH --output=/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/logs/Resynthesis_styleclip_DCTFO%j.log   # STDOUT/STDERR

# Load conda environment
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/a_shahj/defake

# Make sure logs directory exists
mkdir -p /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/logs

# Training parameters
GPU_ID=0  # GPU index (SLURM will assign, but you can override)
INIT_LR=1e-2
EPOCH=100
BATCH_SIZE=24
INPUT_CHANNEL=512

LR_RECONSTRUCTION=1e-3
LOSS_WEIGHT=1

SR_RESUME='./pretrained_models/stylegan_celeba_stage5_noising/sr.pth.tar'

OUTPUT_PATH='/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/saved_model_DCTFO'

DATA_ROOT_POS='/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/StyleCLIP_dataset/train/real'
DATA_ROOT_NEG='/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/StyleCLIP_dataset/train/fake'


# Run training
python train.py -a resnet50 \
  --gpu ${GPU_ID} \
  --batch-size ${BATCH_SIZE} \
  --lr ${INIT_LR} \
  --epochs ${EPOCH} \
  --data-root-pos ${DATA_ROOT_POS} \
  --data-root-neg ${DATA_ROOT_NEG} \
  --input-channel ${INPUT_CHANNEL} \
  --sr-weights-file ${SR_RESUME} \
  --output-path ${OUTPUT_PATH} \
  --lr-sr ${LR_RECONSTRUCTION} \
  --lw-sr ${LOSS_WEIGHT} \
  --no_dilation \
  --sr-scale 4 \
  --sr-num-features 64 \
  --sr-growth-rate 64 \
  --sr-num-blocks 16 \
  --sr-num-layers 8 \
  --idx-stages 5 \
  --mode-sr denoising
