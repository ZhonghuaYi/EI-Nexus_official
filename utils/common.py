"""
Common functions for experiments.
"""


import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda
import numpy as np
import random
import wandb
from omegaconf import DictConfig, OmegaConf

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(seed: int, cudnn_enabled: bool, allow_tf32: bool, num_threads: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if cudnn_enabled:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    
    if allow_tf32:
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
    
    # torch.set_num_threads(num_threads)
    
    
def parallel_model(model, device, rank, local_rank):
    # DDP mode
    ddp_mode = device.type != 'cpu' and rank != -1
    if ddp_mode:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_envs():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size


def set_cuda_devices(gpus):
    """
    Set the CUDA_VISIBLE_DEVICES environment variable.
    """
    if isinstance(gpus, list):
        devices = ''
        for i in range(len(gpus)):
            if i > 0:
                devices += ", "
            devices += str(gpus[i])
            
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.set_device(f'cuda:{gpus}')

        
def padding_tensor(tensor, p):
    h, w = tensor.shape[-2:]
    
    h_padding0, h_padding1 = (p - h % p) // 2, (p - h % p) // 2 + (p - h % p) % 2
    w_padding0, w_padding1 = (p - w % p) // 2, (p - w % p) // 2 + (p - w % p) % 2
    tensor = torch.nn.functional.pad(tensor, (w_padding0, w_padding1, h_padding0, h_padding1))
    
    return tensor


class Padder:
    def __init__(self, shape, p):
        self.shape = shape
        self.p = p
        h, w = shape[-2:]
        h_padding0, h_padding1 = (p - h % p) // 2, (p - h % p) // 2 + (p - h % p) % 2
        w_padding0, w_padding1 = (p - w % p) // 2, (p - w % p) // 2 + (p - w % p) % 2
        self.padding_size = (w_padding0, w_padding1, h_padding0, h_padding1)
    
    def pad(self, *args):
        """
        Pad the input tensors to the same size.
        Args:
            *args (torch.Tensor): tensors to pad.
        Returns:
            padded_args (list): list of padded tensors.
        """
        out = []
        for arg in args:
            if arg.dtype == torch.bool:
                out.append(F.pad(arg, self.padding_size, mode='constant'))
            else:
                out.append(F.pad(arg, self.padding_size, mode='replicate'))
        
        return out
    
    def unpad(self, *args):
        """
        Unpad the input tensors to the same size.
        Args:
            *args (torch.Tensor): tensors to unpad.
        Returns:
            unpadded_args (list): list of unpadded tensors.
        """
        
        out = []
        
        for arg in args:
            h, w = arg.shape[-2:]
            c = [self.padding_size[2], h - self.padding_size[3], self.padding_size[0], w - self.padding_size[1]]
            out.append(arg[..., c[0]:c[1], c[2]:c[3]].clone().contiguous())
        
        return out


if __name__ == '__main__':
    a = torch.randint(0, 10, (1, 1, 2, 3)) > 5
    print(a)
    a = padding(4, a)
    print(a)
