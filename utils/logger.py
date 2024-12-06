from typing import List

import logging
import os
import datetime
from shutil import copyfile, copytree
from rich import pretty, print

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, exp_name: str, status_freq: int=100, files_to_backup: List=None, dirs_to_backup: List=None, mode='train'):
        """
        Args:
            exp_name (str): experiment name
            status_freq (int): frequency to print training status
            files_to_backup (List): list of files to backup
            dirs_to_backup (List): list of dirs to backup
        """
        self.exp_name = exp_name
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        if mode == 'train':
            log_dir = os.path.join('runs', current_time + '_' + exp_name)
        elif mode == 'test':
            log_dir = os.path.join('runs', 'test', current_time + '_' + exp_name)
        else:
            raise ValueError(f'Unsupported mode: {mode}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
            print(f"[bold yellow]Log directory: {log_dir}[/bold yellow]")
        
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        self.logger = self._init_logger()
        
        for file in files_to_backup:
            copyfile(file, f"{self.log_dir}/{file}")
            
        for dir in dirs_to_backup:
            if not os.path.exists(f"{self.log_dir}/{dir}"):
                copytree(dir, f"{self.log_dir}/{dir}")
        
        self.total_steps = 0
        self.running_status = {}
        self.status_freq = status_freq
        
        pretty.install()

    def _print_training_status(self):
        """Print training status."""

        for k in self.running_status:
            self.tb_writer.add_scalar(k, self.running_status[k] / self.status_freq, self.total_steps)
            self.running_status[k] = 0.0
            
    def _init_logger(self):
        """Initialize logger. Note that if you use hydra, the hydra logger will be initialized automatically, so that you will have two loggers."""
        logger = logging.getLogger()
        logger.removeHandler(logger.handlers[0])  # remove the default StreamHandler
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.log_dir, f'train_{self.exp_name}.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
    
    def log_info(self, info, rank=0, cmd_print=True):
        """Print information."""
        if rank in [-1, 0]:
            self.logger.info(info)
            if cmd_print:
                print(info)

    def write_status(self, metrics):
        """Push training status."""
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_status:
                self.running_status[key] = 0.0

            self.running_status[key] += metrics[key]

        if self.total_steps % self.status_freq == self.status_freq - 1:
            self._print_training_status()
            self.running_status = {}

    def write_results(self, results):
        """Write results to tensorboard."""
        self.log_info(results)
        
        if self.tb_writer is None:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.tb_writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        """Close tensorboard writer."""
        self.tb_writer.close()
