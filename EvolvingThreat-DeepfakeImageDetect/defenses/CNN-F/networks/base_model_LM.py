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
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        '''
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }
        '''
        # Hybrid model
        
        state_dict = {
            'model_cnn': self.model_cnn.state_dict(),
            'model_mlp': self.model_mlp.state_dict(),
            'final_fc': self.final_fc.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }
        
        
        '''
        # DCT only
        state_dict = {
            'model_mlp': self.model_mlp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }
        '''

        torch.save(state_dict, save_path)

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
