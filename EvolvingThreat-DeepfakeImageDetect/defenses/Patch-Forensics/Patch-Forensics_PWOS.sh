#!/encs/bin/tcsh
#SBATCH --job-name=Patch-Forensics_PWOS                # Job name
#SBATCH --mail-type=ALL                                # Email notifications
#SBATCH --chdir=/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics   # Working directory
#SBATCH --nodes=1                                      # Number of nodes
#SBATCH --ntasks=1                                     # Number of tasks
#SBATCH --cpus-per-task=16                             # CPU cores per task
#SBATCH --mem=256G                                     # Memory per node
#SBATCH --time=168:00:00                               # Runtime (7 days)
#SBATCH --account=nebular                              # Your valid account
#SBATCH --partition=cl                                 # Partition containing A100 GPUs
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1               # Request 1 × A100 80GB GPU

#SBATCH --output=/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/logs/Patch-Forensics_PWOS_train_%j.log

# ==============================
# Setup environment
# ==============================

# Load conda (tcsh)
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.csh

# Activate the correct environment
conda activate /speed-scratch/a_shahj/defake

# Ensure logs directory exists
mkdir -p /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/logs

# ==============================
# Run training
# ==============================
python3 train.py checkpoints/gp2-faceforensics-df_seed0_xception_block2_constant_p20_V1/opt.yml \
    --real_im_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/real \
    --fake_im_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/fake \
    --load_model \
    --which_epoch bestval \
    --overwrite_config \
    --batch_size 48 \
    --patch_shuffle \
    --patch_size 16 \
    --patch_shuffle_prob 0.5 
