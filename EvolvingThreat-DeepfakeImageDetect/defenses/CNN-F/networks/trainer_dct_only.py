import torch
import torch.nn as nn
from networks.base_model import BaseModel

import torch
import torch.nn as nn
from networks.base_model import BaseModel


# =========================================
# ADD THIS BLOCK RIGHT HERE (below imports)
# =========================================
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
        hidden = 512
        self.fc_in = nn.Linear(input_dim, hidden)
        self.act = nn.SiLU()            # ✅ instantiate once
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
        x = self.act(self.fc_in(x))     # ✅ call instance, not class
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.final(x)

# =========================================
# END OF NEW BLOCK
# =========================================



# -----------------------------
# MLP for DCT stats
# -----------------------------
class DCTStatMLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 128))  # final DCT embedding
        layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [B, 32]


# -----------------------------
# Trainer: DCT MLP only
# -----------------------------
class Trainer(BaseModel):
    def name(self):
        return 'DCTTrainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)



        self.model_mlp = DCTClassifier(input_dim=81920)
        dct_out_dim = 1   # classifier already outputs logits


        # ---- Move to GPU ----
        self.model_mlp.to(opt.gpu_ids[0])

        # ---- Loss and optimizer ----
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            params = list(self.model_mlp.parameters())

            if opt.optim == 'adam':
                self.optimizer = torch.optim.AdamW(params, lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)


            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9)
            else:
                raise ValueError("optim should be [adam, sgd]")

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )

        # ---- Load pretrained weights if needed ----
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch, opt.modeltype)

    # ---- Prepare inputs ----
    def set_input(self, input):
        self.dct = input[1].float().to(self.device)
        self.label = input[2].float().unsqueeze(1).to(self.device)


    # ---- Forward pass ----
    def forward(self, dct):
        return self.model_mlp(dct)   # already outputs logits


    # ---- Loss calculation ----
    def get_loss(self):
        return self.loss_fn(self.output, self.label)

    # ---- Optimization ----
    def optimize_parameters(self):
        self.output = self.forward(self.dct)
        self.loss = self.loss_fn(self.output, self.label)

        self.optimizer.zero_grad()
        self.loss.backward()

        # ---- Add gradient clipping ----
        torch.nn.utils.clip_grad_norm_(self.model_mlp.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.model_mlp.parameters(), 5.0)


        self.optimizer.step()


    # ---- Update LR scheduler ----
    def update_scheduler(self, val_acc):
        if self.isTrain and hasattr(self, 'scheduler'):
            self.scheduler.step(val_acc)

    # ---- Adjust learning rate manually ----
    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True
