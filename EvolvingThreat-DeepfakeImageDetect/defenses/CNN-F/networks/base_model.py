# from pix2pix
import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    
    def save_networks(self, epoch):
        save_filename = f"model_epoch_{epoch}.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        # ------------------------------------------------------------
        # Save model parameters
        # ------------------------------------------------------------
        state_dict = {
            'model_cnn': self.model_cnn.state_dict(),
            'model_mlp': self.model_mlp.state_dict(),
            'final_fc': self.final_fc.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }

        # ------------------------------------------------------------
        # Save DCT mean/std (safe + supports ConcatDataset)
        # ------------------------------------------------------------
        if hasattr(self, "train_dataset") and self.train_dataset is not None:

            dct_means = []
            dct_stds = []

            # Case 1: ConcatDataset → iterate through sub-datasets
            if isinstance(self.train_dataset, torch.utils.data.ConcatDataset):
                for ds in self.train_dataset.datasets:
                    if hasattr(ds, "get_dct_mean") and hasattr(ds, "get_dct_std"):
                        try:
                            dct_means.append(ds.get_dct_mean())
                            dct_stds.append(ds.get_dct_std())
                        except RuntimeError:
                            print("=> Warning: Some sub-dataset DCT stats not ready yet.")
                    else:
                        print("=> Warning: Sub-dataset has no DCT stat functions.")

                if len(dct_means) > 0:
                    dct_mean = torch.stack(dct_means, dim=0).mean(dim=0)
                    dct_std  = torch.stack(dct_stds, dim=0).mean(dim=0)
                    state_dict["dct_mean"] = dct_mean.cpu()
                    state_dict["dct_std"]  = dct_std.cpu()
                    print("=> Included (averaged) DCT mean/std from ConcatDataset")
                else:
                    print("=> Warning: No DCT stats found in ConcatDataset. Skipping...")

            # Case 2: Single Dataset
            else:
                if hasattr(self.train_dataset, "get_dct_mean") and hasattr(self.train_dataset, "get_dct_std"):
                    try:
                        dct_mean = self.train_dataset.get_dct_mean()
                        dct_std  = self.train_dataset.get_dct_std()
                        state_dict["dct_mean"] = dct_mean.cpu()
                        state_dict["dct_std"]  = dct_std.cpu()
                        print("=> Included DCT mean/std from dataset")
                    except RuntimeError:
                        print("=> Warning: DCT stats not ready yet (dataset not iterated).")
                else:
                    print("=> Dataset has no DCT stat methods. Skipping...")

        else:
            print("=> No train_dataset attached, DCT stats NOT saved")

        # ------------------------------------------------------------
        # Final save
        # ------------------------------------------------------------
        torch.save(state_dict, save_path)
        print(f"=> Saved checkpoint to {save_path}")


   

    # load models from the disk
    
    def load_networks(self, epoch, modeltype):
        """
        Safer loading:
        - During *testing*: skip loading unless modeltype is 0.1 or 0.5
        - During *training*: keep original behavior
        """

        # --- TEST MODE (infer_dct.py) ---
        if not self.isTrain:
            if modeltype not in ["0.1", "0.5"]:
                print(f"[load_networks] Test mode: skipping auto-load (modeltype='{modeltype}')")
                return

        # --- TRAIN MODE (train.py) ---
        if modeltype == "0.1":
            load_path = "./weights/blur_jpg_prob0.1.pth"
        elif modeltype == "0.5":
            load_path = "./weights/blur_jpg_prob0.5.pth"
        else:
            # Training: no pre-trained weights available
            print(f"[load_networks] No pretrained weights for modeltype='{modeltype}', starting from scratch.")
            return

        print(f"loading the model from {load_path}")
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        #self.model.load_state_dict(state_dict['model'])
        self.model_cnn.load_state_dict(state_dict['model_cnn'])
        self.model_mlp.load_state_dict(state_dict['model_mlp'])
        self.final_fc.load_state_dict(state_dict['final_fc'])

        self.total_steps = state_dict['total_steps']
        # ============================================================
        #        RESTORE DCT MEAN / STD  (INSERT HERE)
        # ============================================================
        self.dct_mean = None
        self.dct_std = None

        if "dct_mean" in state_dict:
            self.dct_mean = state_dict["dct_mean"].to(self.device)
            print("=> Loaded DCT mean from checkpoint")

        if "dct_std" in state_dict:
            self.dct_std = state_dict["dct_std"].to(self.device)
            print("=> Loaded DCT std from checkpoint")
        # ============================================================

        if self.isTrain and not self.opt.new_optim:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            ### move optimizer state to GPU
            #for state in self.optimizer.state.values():
                #for k, v in state.items():
                    #if torch.is_tensor(v):
                        #state[k] = v.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        if k == "step":  
                            # keep step counter on CPU
                            state[k] = v.cpu()
                        else:
                            state[k] = v.to(self.device)


            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr

    #def eval(self):
        #self.model.eval()
    def eval(self):
        self.model_cnn.eval()
        self.model_mlp.eval()
        self.final_fc.eval()

    '''
    def test(self):
        with torch.no_grad():
            self.forward()
    '''
    def test(self):
        self.eval()  # set all submodules to eval mode
        with torch.no_grad():
            self.output = self.forward(self.dct)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
