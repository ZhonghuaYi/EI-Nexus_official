import torch
import torch.nn as nn 

from omegaconf import DictConfig


def build_optimizer(config: DictConfig, params: nn.parameter.Parameter) -> torch.optim.Optimizer:
    """
    Build optimizer from config.
    Args:
        config: DictConfig of training config
        model: model to optimize
    Returns:
        optimizer: torch optimizer
    """
    optimizer_type = config.type
    
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            params,
            lr=config.Adam.lr,
            weight_decay=config.Adam.weight_decay,
            amsgrad=config.Adam.amsgrad,
            betas=config.Adam.betas,
            eps=config.Adam.eps,
        )
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            lr=config.AdamW.lr,
            weight_decay=config.AdamW.weight_decay,
            amsgrad=config.AdamW.amsgrad,
            betas=config.AdamW.betas,
            eps=config.AdamW.eps,
        )
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(
            params,
            lr=config.SGD.lr,
            momentum=config.SGD.momentum,
            weight_decay=config.SGD.weight_decay,
            dampening=config.SGD.dampening,
            nesterov=config.SGD.nesterov,
        )
    else:
        raise ValueError(f'Unsupported optimizer type: {optimizer_type}')

    return optimizer
