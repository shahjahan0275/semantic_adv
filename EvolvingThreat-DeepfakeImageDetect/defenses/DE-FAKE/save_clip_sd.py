import torch
import clip

model, _ = clip.load("ViT-B/32", device="cpu")
torch.save(model.state_dict(),
           "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/checkpoints/finetune_clip.pt")
print("Saved finetune_clip.pt successfully!")
