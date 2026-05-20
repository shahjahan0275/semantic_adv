# [On Improving Robustness of Deepfake  Image Detectors] [USENIX Security ‘26]

This repository contains the official PyTorch implementation of the Robustness of Deepfake  Image Detectors authored by Abu Taib Mohammed Shahjahan, Mohammad Mannan, A. Ben Hamza and Amr Youssef. If you discover our code to be valuable for your research, kindly consider including the following citation:

 ```
@article{shahjahan2026Deepfake,
   title={On Improving Robustness of Deepfake  Image Detectors},
   author={Shahjahan, Abu Taib Mohammed and Mannan,Mohammad and Hamza, A Ben and Youssef, Amr },
   conference={USENIX Security ‘26},
   year={2026},      
 }
```
### Important Note About File Paths

The codebase was developed and evaluated across multiple HPC clusters and local machines. Therefore, some dataset paths, checkpoint paths, and output directories in the source code may still contain the original absolute paths used during our experiments.

Please update these paths according to your local system or computing environment before running the code.

### Important Note About sample data
We have uploaded a few samples in the folder "sample_data", which could be used to run the code but not enough to reproduce our result . All the data that we used in our work could be downloaded from the original author's repository as described in the paper.

## D3_imp
### Network Architecture

<div align="center">
  <img src="https://github.com/shahjahan0275/semantic_adv/blob/main/demo/Architecture.png?raw=true" width="100%">
</div>


The PyTorch implementation for Improved D3

#### Reproducing Results
##### Reproducing Table 3

To reproduce the results reported in Table 3 of the paper using the proposed D3_Improved detector:

For inference/evaluation datasets, we used the datasets from Section 5.2 of:

Abdullah et al., "An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape" (IEEE S&P 2024).

For training, we used:

The GenImage dataset, and

The datasets from Abdullah et al. (2024),

as described in the paper.

##### Reproducing Table 2

To reproduce the results in Table 2 using all datasets from Abdullah et al. (2024), please follow the instructions from the original D3 repository:

D3: Scaling Up Deepfake Detection by Learning from Discrepancy (CVPR 2025)

##### Reproducing Table 1

To reproduce the results in Table 1 using all the dataset from Abdullah et al. (2024) please follow the setup and instructions provided in the corresponding repositories of the respective detectors evaluated in our paper.We have provided code link for all the detectors evaluated in this table in the acknowledgement section of this README page.


## Environment Setup

### Option 1 (Recommended): Conda Environment

```bash
conda env create -f environment.yml
conda activate D3
```

### Option 2: Pip Installation

Create a new Python 3.10 environment first:

```bash
conda create -n D3 python=3.10
conda activate D3
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```



### Training_Command
1. Only with StyleCLIP Dataset (Sifat2024evolvingthreat)
   
python train_3branch_FOS_F.py \
  --name train_d3B_FO4_DCT_StyleCLIP_ReRUN \
  --train_samples 16000 \
  --arch CLIP:ViT-L/14 \
  --checkpoints_dir /forcolab/home/ashahj/D3/ckpt \
  --fix_backbone \
  --head_type attention \
  --batch_size 32 \
  --shuffle \
  --patch_size 14 \
  2>&1 | tee logs/train_$(date +%F_%H-%M).log

2. StyleCLIP Dataset (Sifat2024evolvingthreat) and Genimage(Zhu2024genimage)
   
python train_3branch_FOS_F.py \
  --name train_d3B_FO4_DCT_StyleCLIP_genimage_ReRUN \
  --train_samples 36000 \
  --arch CLIP:ViT-L/14 \
  --checkpoints_dir /forcolab/home/ashahj/D3/ckpt \
  --fix_backbone \
  --head_type attention \
  --batch_size 32 \
  --shuffle \
  --patch_size 14 \
  2>&1 | tee logs/train_$(date +%F_%H-%M).log

### Inference_Command
python validate_for_robustness_spd.py

## Reproducing Table 4 Results

The detectors located inside the defenses/ folder are used to reproduce the results reported in Table 4 of the paper.

Detailed setup, preprocessing, and usage instructions for these detectors can be found in:

The repository accompanying Abdullah et al., "An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape" (IEEE S&P 2024), and

The original repositories of the respective detectors.

Please follow the corresponding detector-specific instructions for downloading pretrained weights, preparing datasets, and running inference/training pipelines.

Some dataset paths and checkpoint paths in our code may still reflect the original development environments (HPC clusters and local machines). Update these paths according to your system configuration before execution.

## Environment Setup

The following improved detectors by us CNN-F,Resynthesis,DCT,DE-FAKE and  Patch-Forensics were tested with:

- Python 3.8
- PyTorch 1.12.1
- CUDA 11.3

### Option 1 (Recommended): Conda Environment

Create the environment directly from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate defake
```

### Option 2: Pip Installation (Fallback)

If Conda environment creation fails on your system, create a clean Python environment first:

```bash
conda create -n defake python=3.8
conda activate defake
```

Then install dependencies using:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```

Expected output:

```bash
1.12.1
```

### GPU Check (Optional)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```bash
True
```


## CNN-F
### Training_Command
python train_dctstats.py --name DCT_ResNet50_LM12 \
--blur_prob 0.1 --blur_sig 0.0,3.0 \
--jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 \
--dataroot /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/mydataset \
--classes general --gpu_ids 0 --modeltype 0.1 --batch_size 64

### Inference_Command
python infer_dct_LM1.py --dir /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet/ --model_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/checkpoints/DCT_ResNet50_LM12/model_epoch_latest.pth

python infer_dct_LM1.py --dir /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/EfficientNet/ --model_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/checkpoints/DCT_ResNet50_LM12/model_epoch_latest.pth

python infer_dct_LM1.py --dir /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/ViT/ --model_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/checkpoints/DCT_ResNet50_LM12/model_epoch_latest.pth

### To reproduce the Figure 6 run

python plot_tsne.py
python plot_tsne_hybrid_DCTstats.py

## Resynthesis
### Network Architecture

<div align="center">
  <img src="https://github.com/shahjahan0275/semantic_adv/blob/main/demo/Re-Synthesis.png?raw=true" width="100%">
</div>


The PyTorch implementation for Improved Resynthesis

Training_Command in the cluster environment with "sbatch". You can modify the command to use as per your hardware requirement.
####################  Resynthesis_styleclip.sh  DCT Fourth-Order Statistics #############################
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

#Load conda environment
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/a_shahj/defake

#Make sure logs directory exists
mkdir -p /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/logs

#Training parameters
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


### Run training
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


### Inference_Command
python infer_dct.py -a resnet50 --gpu 0 --data-root-pos "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet/0_real" --data-root-neg "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet/1_fake" --input-channel 512 --resume "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/saved_model_DCTFO/0100.pth.tar" --sr-weights-file "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/saved_model_DCTFO/0100_sr.pth.tar" --save_path "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/results" --no_dilation --sr-scale 4 --sr-num-features 64 --sr-growth-rate 64 --sr-num-blocks 16 --sr-num-layers 8 --idx-stages 5

python infer_dct.py -a resnet50 --gpu 0 --data-root-pos "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/EfficientNet/0_real" --data-root-neg "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/EfficientNet/1_fake" --input-channel 512 --resume "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/saved_model_DCTFO/0100.pth.tar" --sr-weights-file "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/saved_model_DCTFO/0100_sr.pth.tar" --save_path "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/results" --no_dilation --sr-scale 4 --sr-num-features 64 --sr-growth-rate 64 --sr-num-blocks 16 --sr-num-layers 8 --idx-stages 5

python infer_dct.py -a resnet50 --gpu 0 --data-root-pos "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/ViT/0_real" --data-root-neg "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/ViT/1_fake" --input-channel 512 --resume "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/saved_model_DCTFO/0100.pth.tar" --sr-weights-file "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/saved_model_DCTFO/0100_sr.pth.tar" --save_path "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Resynthesis/results" --no_dilation --sr-scale 4 --sr-num-features 64 --sr-growth-rate 64 --sr-num-blocks 16 --sr-num-layers 8 --idx-stages 5

### To reproduce the Figure 5 run 

python plot_tsne.py

## DCT
For Training like ours please use the "StyleCLIP dataset" of Abdullah et al.~\cite{Sifat2024evolvingthreat}. Download the dataset and place it in the right folder. Due to the space limitations we can't upload the data in our repository, we just kept 3 or 4 training /test images as an example or place holder.
### Training_Command
python train_exp_DS_GI_Frequency_FO_batch.py \
  --train_root /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/StyleCLIP_dataset/train \
  --val_root /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/StyleCLIP_dataset/val \
  --input_size 1024 \
  --num_real_train 144000 \
  --num_fake_train 144000 \
  --num_real_val 14400 \
  --num_fake_val 14400 \
  --epochs 20 \
  --lr 1e-2 \
  --weight_decay 1e-3 \
  --save_dir /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/checkpoints/exp_DS_GI_DCTFO_StyleCLIP


### Inference_Command
python test_exp_DS_GI_Frequency_FO.py --fake_root /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet/1_fake --real_root /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/MidStyleCLIPjourney_train/StyleCLIP_dataset/test/0_real --model_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/checkpoints/exp_DS_GI_DCTFO_StyleCLIP/best_model_MjStyle.pth --meanstd_dir /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/checkpoints/exp_DS_GI_DCTFO_StyleCLIP --out_csv /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/result/CLIPResNet_StyleCLIP_TO.csv

python test_exp_DS_GI_Frequency_FO.py --fake_root /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/EfficientNet/1_fake --real_root /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/MidStyleCLIPjourney_train/StyleCLIP_dataset/test/0_real --model_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/checkpoints/exp_DS_GI_DCTFO_StyleCLIP/best_model_MjStyle.pth --meanstd_dir /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/checkpoints/exp_DS_GI_DCTFO_StyleCLIP --out_csv /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/result/EfficientNet_StyleCLIP_TO.csv

python test_exp_DS_GI_Frequency_FO.py --fake_root /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/ViT/1_fake --real_root /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/MidStyleCLIPjourney_train/StyleCLIP_dataset/test/0_real --model_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/checkpoints/exp_DS_GI_DCTFO_StyleCLIP/best_model_MjStyle.pth --meanstd_dir /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/checkpoints/exp_DS_GI_DCTFO_StyleCLIP --out_csv /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/result/ViT_StyleCLIP_TO.csv

## DE-FAKE
### Network Architecture

<div align="center">
  <img src="https://github.com/shahjahan0275/semantic_adv/blob/main/demo/de-fake.png?raw=true" width="100%">
</div>


The PyTorch implementation for Improved DE-FAKE
### Training_Command
python train_FOS_3BDCT_patch_text.py --epoch 200 --lr 5e-5 --inputpath_linear /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/checkpoints/clip_linear.pt --inputpath_clip /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/checkpoints/finetune_clip.pt --outputpath_linear /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/ckpt_caption_FOS_DCT_patch_text/StyleCLIP_linear_finetuned.pt --outputpath_clip /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/ckpt_caption_FOS_DCT_patch_text/StyleCLIP_clip_finetuned.pt

### Inference_Command
python test_FOS_3BDCT_patch_text.py --outputpath_clip /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/ckpt_caption_FOS_DCT_patch_text/StyleCLIP_clip_finetuned.pt --outputpath_linear /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/ckpt_caption_FOS_DCT_patch_text/StyleCLIP_linear_finetuned.pt

## Patch-Forensics
### Network Architecture

<div align="center">
  <img src="https://github.com/shahjahan0275/semantic_adv/blob/main/demo/patch-forensics.png?raw=true" width="100%">
</div>


The PyTorch implementation for Improved Patch-Forensics
For Training like ours please use the "StyleCLIP dataset" of Abdullah et al.~\cite{Sifat2024evolvingthreat}. Download the dataset and place it in the right folder. Due to the space limitations we can't upload the data in our repository, we just kept 3 or four training /test images as an example or place holder.
### Training_Command

python3 train_V1.py checkpoints/gp2-faceforensics-df_seed0_xception_block2_constant_p20_V1/opt.yml \
    --real_im_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/real \
    --fake_im_path /speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/fake \
    --load_model \
    --which_epoch bestval \
    --overwrite_config \
    --batch_size 32 \
    --patch_shuffle \
    --patch_size 16 \
    --patch_shuffle_prob 0.5
    
### Inference_Command
python3 test_dct.py --gpu_ids 0 --which_epoch bestval --partition test --dataset_name CLIPResNet --real_im_path /media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/real/test --fake_im_path /media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/fake/test --train_config checkpoints/gp2-faceforensics-df_seed0_xception_block2_constant_p20_V1/opt.yml

python3 test_dct.py --gpu_ids 0 --which_epoch bestval --partition test --dataset_name EfficientNet --real_im_path /media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/real/test --fake_im_path /media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/fake/test --train_config checkpoints/gp2-faceforensics-df_seed0_xception_block2_constant_p20_V1/opt.yml

python3 test_dct.py --gpu_ids 0 --which_epoch bestval --partition test --dataset_name ViT --real_im_path /media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/real/test --fake_im_path /media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/Patch-Forensics/mydataset/fake/test --train_config checkpoints/gp2-faceforensics-df_seed0_xception_block2_constant_p20_V1/opt.yml

## Evaluating Our Pre-trained Models

The pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1CCixsANcVFWiLvUMJtuPXcK2y5NUJHws). Put them in the respective directory.

## Acknowledgement

Our code makes references to the following repositories.
- [D3](https://github.com/BigAandSmallq/D3).
- [EvolvingThreat](https://github.com/secml-lab-vt/EvolvingThreat-DeepfakeImageDetect).
- [GenImage](https://github.com/GenImage-Dataset/GenImage).
- [patch-forensics](https://github.com/chail/patch-forensics).
- [CNN-F](https://github.com/PeterWang512/CNNDetection).
- [De-Fake](https://github.com/zeyangsha/De-Fake).
- [DCT](https://github.com/jonasricker/diffusion-model-deepfake-detection).
- [Resynthesis](https://github.com/SSAW14/BeyondtheSpectrum).
- [spai](https://github.com/mever-team/spai).
- [CO-SPY](https://github.com/Megum1/Co-Spy).
- [FIRE](https://github.com/Chuchad/FIRE).
- [ManifoldBias](https://github.com/JonathanBrok/Manifold-Induced-Biases-for-Zero-shot-and-Few-shot-Detection-of-Generated-Images).
- [Directionality](https://github.com/uibk-uncover/directionality).
- [Upsampling](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).

We thank the authors for sharing their code and kindly request that you also acknowledge their contributions by citing their work.

## References

[1] Yang, Yongqi, et al.
    "D3: Scaling Up Deepfake Detection by Learning from Discrepancy."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[2] Abdullah, Sifat Muhammad, et al.
    "An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape."
    IEEE Symposium on Security and Privacy (SP), 2024.

[3] Zhu, Mingjian, et al.
    "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image."
    Advances in Neural Information Processing Systems (NeurIPS), 2024.

