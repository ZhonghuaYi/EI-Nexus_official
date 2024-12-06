import torch
import torch.nn as nn 

from omegaconf import DictConfig


def build_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build scheduler from config.
    Args:
        config: DictConfig of training config
        optimizer: torch optimizer
    Returns:
        scheduler: torch scheduler
    """
    scheduler_type = config.type
    
    if scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.StepLR.step_size,
            gamma=config.StepLR.gamma,
            last_epoch=config.StepLR.last_epoch,
        )
    elif scheduler_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.MultiStepLR.milestones,
            gamma=config.MultiStepLR.gamma,
            last_epoch=config.MultiStepLR.last_epoch,
        )
    elif scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.ExponentialLR.gamma,
            last_epoch=config.ExponentialLR.last_epoch,
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.CosineAnnealingLR.T_max,
            eta_min=config.CosineAnnealingLR.eta_min,
            last_epoch=config.CosineAnnealingLR.last_epoch,
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.ReduceLROnPlateau.mode,
            factor=config.ReduceLROnPlateau.factor,
            patience=config.ReduceLROnPlateau.patience,
            threshold=config.ReduceLROnPlateau.threshold,
            threshold_mode=config.ReduceLROnPlateau.threshold_mode,
            cooldown=config.ReduceLROnPlateau.cooldown,
            min_lr=config.ReduceLROnPlateau.min_lr,
            eps=config.ReduceLROnPlateau.eps,
            verbose=config.ReduceLROnPlateau.verbose,
        )
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.CosineAnnealingWarmRestarts.T_0,
            T_mult=config.CosineAnnealingWarmRestarts.T_mult,
            eta_min=config.CosineAnnealingWarmRestarts.eta_min,
            last_epoch=config.CosineAnnealingWarmRestarts.last_epoch,
        )
    elif scheduler_type == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.CyclicLR.base_lr,
            max_lr=config.CyclicLR.max_lr,
            step_size_up=config.CyclicLR.step_size_up,
            step_size_down=config.CyclicLR.step_size_down,
            mode=config.CyclicLR.mode,
            gamma=config.CyclicLR.gamma,
            scale_fn=config.CyclicLR.scale_fn,
            scale_mode=config.CyclicLR.scale_mode,
            cycle_momentum=config.CyclicLR.cycle_momentum,
            base_momentum=config.CyclicLR.base_momentum,
            max_momentum=config.CyclicLR.max_momentum,
            last_epoch=config.CyclicLR.last_epoch,
        )
    elif scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.OneCycleLR.max_lr,
            total_steps=config.OneCycleLR.total_steps,
            epochs=config.OneCycleLR.epochs,
            steps_per_epoch=config.OneCycleLR.steps_per_epoch,
            pct_start=config.OneCycleLR.pct_start,
            anneal_strategy=config.OneCycleLR.anneal_strategy,
            cycle_momentum=config.OneCycleLR.cycle_momentum,
            base_momentum=config.OneCycleLR.base_momentum,
            max_momentum=config.OneCycleLR.max_momentum,
            div_factor=config.OneCycleLR.div_factor,
            final_div_factor=config.OneCycleLR.final_div_factor,
            last_epoch=config.OneCycleLR.last_epoch,
        )
    else:
        raise ValueError(f'Unsupported scheduler type: {scheduler_type}')

    return scheduler
