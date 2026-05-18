#original
'''
class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt)
        print(f"using {self.model.__class__.__name__}")

        if opt.head_type == "fc":
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        elif opt.head_type == "attention":
            for name, params in self.model.attention_head.named_parameters():
                torch.nn.init.normal_(params, 0.0, opt.init_gain)
        
        if opt.resume_path is not None:
            state_dict = torch.load(opt.resume_path)
            if self.opt.fix_backbone:
                if self.opt.head_type == "attention" or opt.head_type == "crossattention":
                    self.model.attention_head.load_state_dict(state_dict)
                else:
                    self.model.fc.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

        if opt.fix_backbone:
            params = []
            if opt.head_type == "fc":
                for name, p in self.model.named_parameters():
                    if  name=="fc.weight" or name=="fc.bias": 
                        params.append(p) 
                    else:
                        p.requires_grad = False
            elif opt.head_type == "mlp":
                for p in self.model.mlp.parameters():
                    params.append(p)
            elif opt.head_type == "attention" or opt.head_type == "crossattention":
                for p in self.model.attention_head.parameters():
                    params.append(p)

            elif opt.head_type == "transformer":
                params = [{'params': self.model.transformer_block.parameters()},
                {'params': self.model.fc.parameters()}]
                # params = self.model.transformer.parameters()
                # params["fc"] = self.model.fc.parameters()


        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.model = nn.parallel.DistributedDataParallel(self.model)
        self.model.to(opt.gpu_ids[0])



    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)
        # self.output = self.output.view(-1).unsqueeze(1)
        self.output = self.output




    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
'''
'''
import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model
import models.clip_models as clip_models

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt)
        print(f"using {self.model.__class__.__name__}")

        if opt.head_type == "fc":
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        elif opt.head_type == "attention":
            for name, params in self.model.attention_head.named_parameters():
                torch.nn.init.normal_(params, 0.0, opt.init_gain)
        
        if opt.resume_path is not None:
            state_dict = torch.load(opt.resume_path)
            if self.opt.fix_backbone:
                if self.opt.head_type == "attention" or opt.head_type == "crossattention":
                    self.model.attention_head.load_state_dict(state_dict)
                else:
                    self.model.fc.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

        if opt.fix_backbone:
            params = []
            if opt.head_type == "fc":
                for name, p in self.model.named_parameters():
                    if  name=="fc.weight" or name=="fc.bias": 
                        params.append(p) 
                    else:
                        p.requires_grad = False
            elif opt.head_type == "mlp":
                for p in self.model.mlp.parameters():
                    params.append(p)
            elif opt.head_type == "attention" or opt.head_type == "crossattention":
                for p in self.model.attention_head.parameters():
                    params.append(p)

            elif opt.head_type == "transformer":
                params = [{'params': self.model.transformer_block.parameters()},
                          {'params': self.model.fc.parameters()}]

        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.model = nn.parallel.DistributedDataParallel(self.model)
        self.model.to(opt.gpu_ids[0])

        # ✅ NEW: keep placeholders for input and features
        self.extra_features = None
        self.output = None
        self.loss = None

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

        # ✅ UPDATED to accept tuple of features
    def set_input(self, input, extra_features=None):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        if extra_features is not None:
            # tuple of (original, denoised, noise_features)
            self.extra_features = tuple(x.to(self.device) for x in extra_features)
        else:
            self.extra_features = None

    def forward(self):
        if self.extra_features is not None:
            original, denoised, noise = self.extra_features
            self.output = self.model((original, denoised, noise))
        else:
            self.output = self.model(self.input)


    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
'''
import os
import time
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model
import models.clip_models as clip_models  # module (not a class)
#from models import clip_models_DCT as clip_models  #For DCT 3 Branch Training
#from models import clip_models_SOS as clip_models   #For Second order statistics
#from models import clip_models_TOS as clip_models   #For Third order statistics


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt

        # Create model (same as before)
        self.model = get_model(opt)
        print(f"using {self.model.__class__.__name__}")

        # initialize heads if necessary (preserve your prior behavior)
        if opt.head_type == "fc":
            if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        elif opt.head_type == "attention":
            if hasattr(self.model, "attention_head"):
                for name, params in self.model.attention_head.named_parameters():
                    try:
                        torch.nn.init.normal_(params, 0.0, opt.init_gain)
                    except Exception:
                        pass

        # optionally load resume
        if opt.resume_path is not None:
            state_dict = torch.load(opt.resume_path, map_location="cpu")
            if self.opt.fix_backbone:
                if self.opt.head_type in ("attention", "crossattention"):
                    self.model.attention_head.load_state_dict(state_dict)
                else:
                    self.model.fc.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

        # build param list depending on fix_backbone
        if opt.fix_backbone:
            params = []
            if opt.head_type == "fc":
                for name, p in self.model.named_parameters():
                    if name == "fc.weight" or name == "fc.bias":
                        params.append(p)
                    else:
                        p.requires_grad = False
            elif opt.head_type == "mlp":
                if hasattr(self.model, "mlp"):
                    for p in self.model.mlp.parameters():
                        params.append(p)
            elif opt.head_type in ("attention", "crossattention"):
                for p in self.model.attention_head.parameters():
                    params.append(p)
            elif opt.head_type == "transformer":
                params = [
                    {'params': self.model.transformer_block.parameters()},
                    {'params': self.model.fc.parameters()}
                ]
            else:
                # fallback: allow training entire model if nothing matched
                params = self.model.parameters()
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            time.sleep(2)
            params = self.model.parameters()

        # optimizer selection (keep your existing logic)
        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        # device: prefer opt.gpu_ids if available (keeps original semantics)
        if hasattr(opt, "gpu_ids") and len(opt.gpu_ids) > 0:
            device_idx = opt.gpu_ids[0]
            if device_idx >= 0 and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{device_idx}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # move model to device
        self.model.to(self.device)

        # bookkeeping fields used by training script
        self.extra_features = None
        self.output = None
        self.loss = None
        self.total_steps = 0

        # === Robust CLIP output-dim inference ===
        self.clip_out_dim = self._infer_clip_out_dim(opt)
        print(f"[Trainer] Using CLIP output dim = {self.clip_out_dim}")

    def _infer_clip_out_dim(self, opt):
        """
        Try to determine the CLIP (penultimate) feature size robustly:
          1) from models.clip_models.CHANNELS mapping (preferred),
          2) from model.fc.in_features (if present),
          3) from model.attention_head heuristics,
          4) fallback to running a small encode_image probe (if possible),
          5) final fallback to 1024.
        """
        # 1) try clip_models.CHANNELS
        try:
            arch_name = opt.arch.split(":", 1)[1] if ":" in opt.arch else opt.arch
        except Exception:
            arch_name = getattr(opt, "arch", "ViT-L/14")

        # prefer penultimate
        penultimate_key = arch_name + "-penultimate"
        if hasattr(clip_models, "CHANNELS"):
            channels = getattr(clip_models, "CHANNELS")
            if penultimate_key in channels:
                return channels[penultimate_key]
            if arch_name in channels:
                return channels[arch_name]

        # 2) try self.model.fc
        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            return int(self.model.fc.in_features)

        # 3) try attention_head attributes heuristics
        if hasattr(self.model, "attention_head"):
            att = self.model.attention_head
            for candidate in ("d_model", "embed_dim", "hidden_dim", "in_dim", "in_features", "last_dim"):
                if hasattr(att, candidate):
                    val = getattr(att, candidate)
                    try:
                        return int(val)
                    except Exception:
                        pass

        # 4) try a small forward probe on the CLIP backbone (safe, no grad)
        try:
            # if model wraps a clip backbone as attribute `model` (common in your repo)
            backbone = None
            if hasattr(self.model, "model"):
                backbone = getattr(self.model, "model")
            elif hasattr(self.model, "clip_backbone"):
                backbone = getattr(self.model, "clip_backbone")

            if backbone is not None and hasattr(backbone, "encode_image"):
                self.model.eval()
                with torch.no_grad():
                    probe = torch.zeros(1, 3, 224, 224, device=self.device)
                    feat = backbone.encode_image(probe)
                    if hasattr(feat, "shape"):
                        return int(feat.shape[-1])
        except Exception:
            # ignore probe failures
            pass

        # 5) final fallback
        return 1024

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    # Keep set_input signature but IGNORE extra_features (Option A)
    def set_input(self, input, extra_features=None):
        """
        input: expects the same structure your dataset returns (original code used input[0]/input[1])
        extra_features: ignored in Option A (model computes denoiser/noise internally)
        """
        # old code: self.input = input[0].to(self.device); self.label = input[1].to(self.device).float()
        # Some dataloaders return a tuple/list: (images, labels) -- preserve compatibility
        if isinstance(input, (list, tuple)):
            images = input[0]
            labels = input[1] if len(input) > 1 else None
        elif isinstance(input, dict):
            # if dataset returns dict-like, try to be flexible
            images = input.get("image", None) or input.get("images", None) or input.get("img", None)
            labels = input.get("label", None) or input.get("labels", None) or input.get("target", None)
        else:
            raise ValueError("Unsupported input type for set_input()")

        if images is None:
            raise ValueError("Couldn't find image tensor in input structure")

        self.input = images.to(self.device)
        if labels is not None:
            self.label = labels.to(self.device).float()
        else:
            self.label = None

        # do not use external extra_features in Option A
        self.extra_features = None

    def forward(self):

        self.output = self.model(self.input)
        return self.output


    def get_loss(self):
        if self.label is None:
            raise RuntimeError("Labels not set in set_input; cannot compute loss.")
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        # forward + backward + step
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        # increment total_steps so external code using it keeps working
        self.total_steps += 1

    # convenience wrappers to keep compatibility with earlier code
    #def save_networks(self, name):
        # simple wrapper: save model state_dict (you can expand to save optimizer/scaler)
        #path = name if os.path.isabs(name) else os.path.join(self.opt.checkpoints_dir, self.opt.name, name)
        #os.makedirs(os.path.dirname(path), exist_ok=True)
        #torch.save(self.model.state_dict(), path)
        #print(f"Saved model -> {path}")
    
    # convenience wrappers to keep compatibility with earlier code
    def save_networks(self, name):
        """Save the full model and optimizer state (including fc, second_order_pool, etc.)"""
        path = name if os.path.isabs(name) else os.path.join(self.opt.checkpoints_dir, self.opt.name, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': getattr(self, 'total_steps', 0),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
        }, path)
        print(f"✅ Saved full model checkpoint -> {path}")
    #def load_networks(self, path):
        #st = torch.load(path, map_location=self.device)
        #self.model.load_state_dict(st)
        #print(f"Loaded model <- {path}")
    '''
    def load_networks(self, path):
        """Load full model checkpoint including optimizer state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Loaded checkpoint <- {path}")
    '''
    # To resume from last save checkpoint
    def load_networks(self, path):
        """Load model checkpoint (supports both old and new formats)."""
        checkpoint = torch.load(path, map_location=self.device)

        # Case 1: new format (dict with model_state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                self.total_steps = checkpoint['epoch']
                print(f"✅ Loaded checkpoint <- {path} (resuming from step {self.total_steps})")
            else:
                print(f"✅ Loaded checkpoint <- {path} (no epoch info found)")
        
        # Case 2: old format (just state_dict)
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Loaded old-format checkpoint <- {path}")
        
        else:
            raise RuntimeError(f"Unrecognized checkpoint format in {path}")

