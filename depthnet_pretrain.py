from __future__ import print_function, division
import logging

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time

from lib.human_depth_loader import DepthHumanDataset
from lib.network_transformer import EVANet
from config.transformer_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.loss import anchor_loss

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torchvision
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = EVANet(self.cfg, with_gs_render=False)
        self.train_set = DepthHumanDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size*2, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = DepthHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, 100100, pct_start=0.01,
                                                       cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.train()
        self.scaler = GradScaler(enabled=True)

    def train(self):
        for i in tqdm(range(self.total_steps, self.cfg.num_steps)):

            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            #  Raft Stereo
            data, flow_loss, metrics = self.model(data)
            loss_anchor = anchor_loss(data)
            loss = flow_loss + + (10**2)*loss_anchor

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                depth_map = data['lmain']['depth_pred']
                image = torchvision.utils.make_grid(depth_map) 
                self.logger.writer.add_image("Depth Train Left", image, global_step=self.total_steps)

                depth_map = data['lmain']['depth']
                image = torchvision.utils.make_grid(depth_map) 
                self.logger.writer.add_image("Depth Origin Left", image, global_step=self.total_steps)

                depth_map = data['rmain']['depth_pred']
                image = torchvision.utils.make_grid(depth_map) 
                self.logger.writer.add_image("Depth Train Right", image, global_step=self.total_steps)

                depth_map = data['rmain']['depth']
                image = torchvision.utils.make_grid(depth_map) 
                self.logger.writer.add_image("Depth Origin Right", image, global_step=self.total_steps)

                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
            metrics.update({
                'anchor': loss_anchor.item(),
            })
            self.logger.push(metrics)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()

            self.total_steps += 1

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        epe_list, one_pix_list = [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)
        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                data, flow_loss, metrics = self.model(data, is_train=False)

                for view in ['lmain', 'rmain']:
                    valid = (data[view]['depth'] != 0.0)

                    epe = torch.sum((data[view]['depth'] - data[view]['depth_pred']) ** 2, dim=1).sqrt()
                    epe = epe.view(-1)[valid.view(-1)]
                    one_per = (epe < 0.01*torch.max(data[view]['depth']))
                    
                    epe_list.append(epe.mean().item())
                    one_per_list.append(one_per.float().mean().item())

        val_epe = np.round(np.mean(np.array(epe_list)), 4)
        val_one_per = np.round(np.mean(np.array(one_per_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): epe {val_epe}, 1per {val_one_per}")
        self.logger.write_dict({'val_epe': val_epe, 'val_1per': val_one_per}, write_step=self.total_steps)
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/pretrain.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
