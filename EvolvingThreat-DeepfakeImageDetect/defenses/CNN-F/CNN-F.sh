#!/encs/bin/tcsh
#SBATCH --job-name=CNN-F                               # Job name
#SBATCH --mail-type=ALL                                # Email notifications
#SBATCH --chdir=/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F   # Working directory
#SBATCH --nodes=1                                      # Number of nodes
#SBATCH --ntasks=1                                     # Number of tasks
#SBATCH --cpus-per-task=16                             # CPU cores per task
#SBATCH --mem=256G                                     # Memory per node
#SBATCH --time=168:00:00                               # Runtime (7 days)
#SBATCH --account=nebular                              # Your valid account
#SBATCH --partition=cl                                 # Partition containing A100 GPUs
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1               # Request 1 × A100 80GB GPU

#SBATCH --output=/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/logs/train_%j.log

# ==============================
# Setup environment
# ==============================

# Load conda (tcsh)
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.csh

# Activate the correct environment
conda activate /speed-scratch/a_shahj/defake

# Ensure logs directory exists
mkdir -p /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/logs

# ==============================
# Run training
# ==============================
python train_dctstats.py \
    --name DCT_ResNet50_LM123 \
    --blur_prob 0.1 --blur_sig 0.0,3.0 \
    --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 \
    --dataroot /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/mydataset \
    --classes general \
    --gpu_ids 0 \
    --modeltype 0.1 \
    --batch_size 64
