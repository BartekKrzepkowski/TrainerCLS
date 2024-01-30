import logging
from collections import defaultdict
from typing import Dict

import torch
from tqdm import tqdm, trange

from src.modules.metrics import RunStatsBiModal
from src.utils.common import LOGGERS_NAME_MAP

from src.utils.utils_trainer import adjust_evaluators, adjust_evaluators_pre_log, create_paths, save_model, load_model
from src.utils.utils_optim import clip_grad_norm


class TrainerClassification:
    def __init__(self, model, criterion, loaders, optim, lr_scheduler, extra_modules, device):
        self.model = model
        self.criterion = criterion
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.logger = None
        self.base_path = None
        self.save_path = None
        self.epoch = -1
        self.global_step = None

        self.extra_modules = extra_modules
        
        
    def run_exp(self, config):        
        self.manual_seed(config)
        self.at_exp_start(config)

        config.kind = 'proper'
        self.run_loop(config.exp_starts_at_epoch, config.exp_ends_at_epoch, config)
        
        step = f'epoch_{self.epoch + 1}'
        save_model(self.model, self.save_path(step))
        self.logger.close()
    
        
    
    def run_loop(self, exp_starts_at_epoch, exp_ends_at_epoch, config):
        """_summary_

        Args:
            exp_starts_at_epoch (_type_): _description_
            exp_ends_at_epoch (_type_): _description_
            config (_type_): _description_
        """
        logging.info('Training started.')
        for epoch in trange(exp_starts_at_epoch, exp_ends_at_epoch, desc='run_exp', leave=True, position=0,
                            colour='green', disable=config.whether_disable_tqdm):
            self.epoch = epoch
            if config.whether_save and epoch % 40 == 0:
                step = f'epoch_{epoch}'
                save_model(self.model, self.save_path(step))
                
            self.model.train()
            self.run_epoch(phase='train', config=config)
            self.model.eval()
            with torch.no_grad():
                self.run_epoch(phase='train_without_aug', config=config)
                self.run_epoch(phase='test', config=config)
                
        logging.info('Training completed.')
            

    def at_exp_start(self, config):
        """
        Initialization of experiment.
        Creates fullname, dirs and logger.
        
        """
        self.base_path, self.save_path = create_paths(config.base_path, config.exp_name)
        config.logger_config['log_dir'] = f'{self.base_path}/{config.logger_config["logger_name"]}'
        self.logger = LOGGERS_NAME_MAP[config.logger_config['logger_name']](config)
        
        self.logger.log_model(self.model, self.criterion, log=None)
        
        if 'stiffness' in self.extra_modules:
            self.extra_modules['stiffness'].logger = self.logger
        if 'hooks_dead_relu' in self.extra_modules:
            self.extra_modules['hooks_dead_relu'].logger = self.logger
        if 'hooks_acts' in self.extra_modules:
            self.extra_modules['hooks_acts'].logger = self.logger
        if 'tunnel' in self.extra_modules:
            self.extra_modules['tunnel'].logger = self.logger
        if 'trace_fim' in self.extra_modules:
            self.extra_modules['trace_fim'].logger = self.logger
        if 'run_stats' in self.extra_modules:
            self.extra_modules['run_stats'].logger = self.logger
            
            
    def run_epoch(self, phase, config):
        """
        Run single epoch
        Args:
            phase (str): phase of the trening
            config (dict):
        """
        logging.info(f'Epoch: {self.epoch}, Phase: {phase}.')
        
        running_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
        }
        epoch_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
        }
        loader_size = len(self.loaders[phase])
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}',
                            leave=False, position=1, total=loader_size, colour='red', disable=config.whether_disable_tqdm)
        self.global_step = self.epoch * loader_size
        
        
        # ════════════════════════ training / inference ════════════════════════ #
        
        
        for i, data in enumerate(progress_bar):
            if phase == 'train_without_aug' and i >= int(loader_size * config.train_without_aug_epoch_freq):
                break
            x_true, y_true = data
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true)
            loss, evaluators = self.criterion(y_pred, y_true)
            step_assets = {
                'evaluators': evaluators,
                'denom': y_true.size(0),
            }
            if 'train' == phase:
                loss.backward()
                if config.clip_value > 0:
                    norm = clip_grad_norm(torch.nn.utils.clip_grad_norm_, self.model, config.clip_value)
                    step_assets['evaluators']['run_stats/model_gradient_norm_squared_from_pytorch'] = norm.item() ** 2
                    
                if self.extra_modules['run_stats'] is not None and config.run_stats_multi and self.global_step % config.run_stats_multi == 0:
                    self.extra_modules['run_stats']('l2', self.global_step)
                    
                self.optim.step()
                
                if self.lr_scheduler is not None and (((self.global_step + 1) % loader_size == 0) or config.logger_config['hyperparameters']['type_names']['scheduler'] != 'multiplicative'):
                    self.lr_scheduler.step()
                    step_assets['evaluators']['lr/training'] = self.optim.param_groups[0]['lr']
                    step_assets['evaluators']['steps/lr'] = self.global_step
                    
                self.optim.zero_grad(set_to_none=True)
                
                if self.extra_modules['trace_fim'] is not None and config.fim_trace_multi and self.global_step % config.fim_trace_multi == 0:
                    self.extra_modules['trace_fim'](self.global_step, config, kind=config.kind)
            
            
            # ════════════════════════ logging ════════════════════════ #
            
            
            running_assets = self.update_assets(running_assets, step_assets, step_assets['denom'], 'running', phase)

            whether_log = (i + 1) % config.log_multi == 0
            whether_epoch_end = (i + 1) == loader_size

            if whether_log or whether_epoch_end:
                epoch_assets = self.update_assets(epoch_assets, running_assets, 1.0, 'epoch', phase)

            if whether_log:
                self.log(running_assets, phase, 'running', progress_bar, self.global_step)
                running_assets['evaluators'] = defaultdict(float)
                running_assets['denom'] = 0.0

            if whether_epoch_end:
                self.log(epoch_assets, phase, 'epoch', progress_bar, self.epoch)

            self.global_step += 1


    def log(self, assets: Dict, phase: str, scope: str, progress_bar: tqdm, step: int):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict):
            phase:
            scope:
            progress_bar:
        '''
        evaluators_log = adjust_evaluators_pre_log(assets['evaluators'], assets['denom'], round_at=4)
        evaluators_log[f'steps/{phase}_{scope}'] = step
        self.logger.log_scalars(evaluators_log, step)
        progress_bar.set_postfix(evaluators_log)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)


    def update_assets(self, assets_target: Dict, assets_source: Dict, multiplier, scope, phase: str):
        '''
        Update epoch assets
        Args:
            assets_target (Dict): Assets to which assets should be transferred
            assets_source (Dict): Assets from which assets should be transferred
            multiplier (int): Number to get rid of the average
            scope (str): Either running or epoch
            phase (str): Phase of the traning
        '''
        assets_target['evaluators'] = adjust_evaluators(assets_target['evaluators'], assets_source['evaluators'],
                                                        multiplier, scope, phase)
        assets_target['denom'] += assets_source['denom']
        return assets_target


    def manual_seed(self, config: defaultdict):
        """
        Set the environment for reproducibility purposes.
        Args:
            config (defaultdict): set of parameters
                usage of:
                    random seed (int):
                    device (torch.device):
        """
        import random
        import numpy as np
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if 'cuda' in self.device.type:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(config.random_seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

