import torch
import torch.nn as nn
import torchvision.models as models

from networks.base_model import BaseModel


# ============================================================
# --- EXACT ORIGINAL ResMLP (NO LayerNorm IN MLP INPUT) ---
# ============================================================

class ResMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)       # <--- THIS IS INSIDE ONLY THE RESBLOCK

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(x)))
        return self.ln(x + h)


# ============================================================
# --- EXACT ORIGINAL DCT CLASSIFIER (MATCH CHECKPOINT) ---
# ============================================================

class DCTClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = 512

        # NO GLOBAL LAYERNORM HERE → matches checkpoint
        self.fc_in = nn.Linear(input_dim, hidden)
        self.act = nn.SiLU()

        self.res1 = ResMLP(hidden)
        self.res2 = ResMLP(hidden)
        self.res3 = ResMLP(hidden)

        self.final = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.act(self.fc_in(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.final(x)


# ============================================================
# ------------ HYBRID TRAINER (LOW MEMORY VERSION) -----------
# ============================================================

class Trainer(BaseModel):

    def name(self):
        return 'HybridTrainer'

    def __init__(self, opt):
        super().__init__(opt)

        # -----------------------------------
        # CNN Encoder (ResNet50)
        # -----------------------------------
        self.model_cnn = models.resnet50(pretrained=True)
        self.model_cnn.fc = nn.Identity()
        cnn_out_dim = 2048

        # -----------------------------------
        # DCT classifier
        # -----------------------------------
        # EXACT DIM YOU USED IN TRAINING
        self.model_mlp = DCTClassifier(input_dim=81920)
        dct_out_dim = 1

        # -----------------------------------
        # Fusion Layer
        # -----------------------------------
        self.final_fc = nn.Sequential(
            nn.Linear(cnn_out_dim + dct_out_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

        # Move to GPU
        self.model_cnn = self.model_cnn.to(opt.gpu_ids[0])
        self.model_mlp = self.model_mlp.to(opt.gpu_ids[0])
        self.final_fc = self.final_fc.to(opt.gpu_ids[0])

        self.scaler = torch.cuda.amp.GradScaler()

        # Optimizer setup
        if self.isTrain:
            params = (
                list(self.model_cnn.parameters()) +
                list(self.model_mlp.parameters()) +
                list(self.final_fc.parameters())
            )

            self.loss_fn = nn.BCEWithLogitsLoss()

            if opt.optim == "adam":
                self.optimizer = torch.optim.Adam(
                    params, lr=opt.lr, betas=(opt.beta1, 0.999)
                )
            else:
                self.optimizer = torch.optim.SGD(
                    params, lr=opt.lr, momentum=0.9
                )

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch, opt.modeltype)

    # ------------------------------------------------------------
    # Input loader
    # ------------------------------------------------------------
    def set_input(self, input):
        self.image = input[0].float().to(self.device)
        self.dct = input[1].float().to(self.device)
        self.label = input[2].float().unsqueeze(1).to(self.device)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, img_batch, dct):
        B, P, C, H, W = img_batch.shape

        assert P == getattr(self.opt, "num_patches", 16), \
            f"Dataset returned {P} patches, expected num_patches={self.opt.num_patches}"

        img_flat = img_batch.reshape(B * P, C, H, W)

        # CNN path
        feats = []
        with torch.cuda.amp.autocast():
            for i in range(0, img_flat.size(0), 1):
                out = self.model_cnn(img_flat[i:i+1])
                feats.append(out)

        flat = torch.cat(feats, dim=0)  # [B*P, 2048]

        splits = torch.split(flat, P, dim=0)
        cnn_feat = torch.stack([s.mean(0) for s in splits], dim=0)

        # DCT path
        dct_feat = self.model_mlp(dct)

        fused = torch.cat([cnn_feat, dct_feat], dim=1)

        with torch.cuda.amp.autocast():
            return self.final_fc(fused)

    # ------------------------------------------------------------
    # Backprop
    # ------------------------------------------------------------
    def optimize_parameters(self):
        with torch.cuda.amp.autocast():
            self.output = self.forward(self.image, self.dct)
            self.loss = self.loss_fn(self.output, self.label)

        self.optimizer.zero_grad()
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def get_loss(self):
        return self.loss_fn(self.output, self.label)
