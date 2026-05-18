# Last modification LM1
import functools
import torch
import torch.nn as nn
import torchvision.models as models
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
from networks.base_model import BaseModel
from torch.nn.functional import binary_cross_entropy_with_logits

# ============================================================
# --- Lightweight ResMLP + DCT classifier ---
# ============================================================
class ResMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(x)))
        return self.ln(x + h)


class DCTClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 1024

        self.norm = nn.LayerNorm(input_dim)
        self.fc_in = nn.Linear(input_dim, hidden)
        self.act = nn.SiLU()

        self.res1 = ResMLP(hidden)
        self.res2 = ResMLP(hidden)
        self.res3 = ResMLP(hidden)

        self.final = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.norm(x)          # normalize input
        x = self.act(self.fc_in(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.final(x)



class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        ce = binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce * (1 - p_t) ** self.gamma
        return loss.mean()




# ============================================================
# ------------ HYBRID TRAINER (LOW MEMORY VERSION) -----------
# ============================================================
class Trainer(BaseModel):
    def name(self):
        return 'HybridTrainer'

    def __init__(self, opt,train_dataset=None ):
        super().__init__(opt)
        self.train_dataset = train_dataset
        self.epoch = 0
        # -----------------------------------
        # CNN: ResNet50 backbone
        # -----------------------------------
        self.model_cnn = models.resnet50(pretrained=True)
        self.model_cnn.fc = nn.Identity()
        cnn_out_dim = 2048

        # -----------------------------------
        # Freeze CNN for first few epochs
        # -----------------------------------
        for p in self.model_cnn.parameters():
            p.requires_grad = False
        self.cnn_frozen_epochs = 0

        # -----------------------------------
        # DCT classifier
        # -----------------------------------
        self.model_mlp = DCTClassifier(input_dim=20480)
        #dct_out_dim = 1


        # -----------------------------------
        # Fusion
        # -----------------------------------
        self.final_fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 128, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )


        # Move models to GPU
        self.model_cnn = self.model_cnn.to(opt.gpu_ids[0])
        self.model_mlp = self.model_mlp.to(opt.gpu_ids[0])
        self.final_fc = self.final_fc.to(opt.gpu_ids[0])

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

        if self.isTrain:
            params = (
                list(self.model_cnn.parameters())
                + list(self.model_mlp.parameters())
                + list(self.final_fc.parameters())
            )

            #self.loss_fn = nn.BCEWithLogitsLoss()
            self.loss_fn = FocalLoss(gamma=2)

            if opt.optim == "adam":
                self.optimizer = torch.optim.Adam(
                    params, lr=opt.lr, betas=(opt.beta1, 0.999)
                )
            else:
                self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9)

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch, opt.modeltype)

    # ------------------------------------------------------------
    # Load batch
    # ------------------------------------------------------------
    def set_input(self, input):
        self.image = input[0].float().to(self.device)  # [B,16,3,224,224]
        self.dct = input[1].float().to(self.device)    # [B,D]
        self.label = input[2].float().unsqueeze(1).to(self.device)

    # ------------------------------------------------------------
    # Forward pass (VERY LOW MEMORY)
    # ------------------------------------------------------------
    def forward(self, img_batch, dct):
        B, P, C, H, W = img_batch.shape
        
        ############  For random patches start###################
        # NEW: Ensure patches per image matches dataset specification
        assert P == getattr(self.opt, "num_patches", 16), \
            f"Dataset returned {P} patches, expected num_patches={self.opt.num_patches}"
        ############  For random patches finish###################
        # flatten patches
        img_flat = img_batch.reshape(B * P, C, H, W)

        # --------------------------------------------------
        # CNN forward (chunk=1 to FORCE minimal memory)
        # --------------------------------------------------
        chunk = 1
        feats = []
        with torch.cuda.amp.autocast():
            for i in range(0, img_flat.size(0), chunk):
                out = self.model_cnn(img_flat[i:i+chunk])
                feats.append(out)

        #cnn_feat = torch.cat(feats, dim=0).reshape(B, P, -1).mean(1)
        # Correct: split feats per sample BEFORE averaging
        flat = torch.cat(feats, dim=0)     # [sum(P_i), 2048]

        # Number of patches per image is fixed = P = img_batch.size(1)
        P = img_batch.size(1)

        # Split into a list of length B
        splits = torch.split(flat, P, dim=0)

        # Now average each sample's patches independently
        cnn_feat = torch.stack([s.mean(0) for s in splits], dim=0)
        if self.epoch <= self.cnn_frozen_epochs:
            cnn_feat = cnn_feat.detach()


        # DCT path
        dct_feat = self.model_mlp(dct)  # [B,128]
        fused = torch.cat([cnn_feat, dct_feat], dim=1)


        with torch.cuda.amp.autocast():
            return self.final_fc(fused)

    # ------------------------------------------------------------
    # Backprop with mixed precision
    # ------------------------------------------------------------
    def optimize_parameters(self):
        # Unfreeze CNN after warm-up
        if self.epoch > self.cnn_frozen_epochs:
            for p in self.model_cnn.parameters():
                p.requires_grad = True

        with torch.cuda.amp.autocast():
            self.output = self.forward(self.image, self.dct)
            self.loss = self.loss_fn(self.output, self.label)

        self.optimizer.zero_grad()
        self.scaler.scale(self.loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()


    # ---- Loss calculation ----
    def get_loss(self):
        return self.loss_fn(self.output, self.label)

    # ---- Update LR scheduler ----
    def update_scheduler(self, val_acc):
        if self.isTrain and hasattr(self, 'scheduler'):
            self.scheduler.step(val_acc)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True





