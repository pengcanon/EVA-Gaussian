from __future__ import print_function, division

import logging

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse

from lib.human_depth_loader import DepthHumanDataset
from lib.network_transformer import EVANet
from config.transformer_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render_feature
from lib.loss import l1_loss, l2_loss, ssim, psnr, opacity_regular, scale_regular, anchor_loss

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

        self.model = EVANet(self.cfg, with_gs_render=True)
        self.train_set = DepthHumanDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size*2, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = DepthHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.val_novel_id = self.cfg.dataset.val_novel_id
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 5000,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            logging.info(f"Using checkpoint from stage1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        
        self.model.train()
        self.scaler = GradScaler(enabled=True)

    def train(self):
        for step in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            data, depth_loss, metrics = self.model.forward(data, is_train=True)
            data, features = pts2render_feature(data, bg_color=self.cfg.dataset.bg_color)

            render_novel_temp = data['novel_view']['img_pred']
            render_novel = self.model.feature_refiner(render_novel_temp, features)

            data['novel_view']['img_refined'] = render_novel
            gt_novel = data['novel_view']['img'].cuda()

            Ll = l2_loss(render_novel[0], gt_novel)
            Lssim = 1.0 - ssim(render_novel[0], gt_novel)
            psnr_value = psnr(render_novel[0], gt_novel).mean().double()
            psnr_temp_value = psnr(render_novel_temp, gt_novel).mean().double()

            L_o_r = opacity_regular(data)
            L_s_r = scale_regular(data)
            L_anchor = anchor_loss(data)
            L1_temp = l2_loss(render_novel_temp,gt_novel)
            Lssim_temp = 1.0 - ssim(render_novel_temp, gt_novel)
            
            loss = depth_loss + 0.8 * (L1_temp+L1) + 0.2 * (Lssim_temp+Lssim) + L_o_r + L_s_r + (10**3)*L_anchor

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                image = torchvision.utils.make_grid(render_novel[0]) 
                self.logger.writer.add_image("Rendered Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(render_novel_temp) 
                self.logger.writer.add_image("Splatted Train", image, global_step=self.total_steps)

                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            metrics.update({
                'l1': Ll.item(),
                'ssim': Lssim.item(),
                'psnr': psnr_value.item(),
            })
            self.logger.push(metrics)

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()

            self.total_steps += 1

        self.model.eval()
        self.run_eval()
        self.model.train()
        
        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))
         

    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        epe_list, one_per_list, psnr_list, ssim_list, lpips_list = [], [], [], [], []
        psnr_list_temp, ssim_list_temp, lpips_list_temp = [], [], []
        psnr_list_refined, ssim_list_refined, lpips_list_refined = [], [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)
        for val_novel_id in self.val_novel_id:
            self.val_set.val_novel_id =  [val_novel_id]
            self.val_loader = DataLoader(self.val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
            self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
            self.val_iterator = iter(self.val_loader)
            for idx in range(self.len_val):
                data = self.fetch_data(phase='val')
                with torch.no_grad():
                    data,_, _ = self.model.forward(data, is_train=False)
                    data, features = pts2render_feature(data, bg_color=self.cfg.dataset.bg_color)
                    
                    render_novel_temp = data['novel_view']['img_pred']
                    render_novel = self.model.feature_refiner(render_novel_temp, features)
                    gt_novel = data['novel_view']['img'].cuda()
                    
                    Lssim = 1.0 - ssim(render_novel[0], gt_novel)
                    psnr_value = psnr(render_novel[0], gt_novel).mean().double()

                    psnr_list.append(psnr_value.item())
                    ssim_list.append(Lssim.mean().item())

                    if idx == show_idx:
                        image = torchvision.utils.make_grid(render_novel[0]) 
                        self.logger.writer.add_image("Rendered Val", image, global_step=self.total_steps)
                        
                        tmp_novel = data['novel_view']['img_pred'][0].detach()
                        tmp_novel *= 255
                        tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                        tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path, self.total_steps)
                        cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

                        image = torchvision.utils.make_grid(gt_novel) 
                        self.logger.writer.add_image("Rendered Val Origin", image, global_step=self.total_steps)

                    for view in ['lmain', 'rmain']:
                        valid = (data[view]['depth'] != 0.0)
                        epe = torch.sum((data[view]['depth'] - data[view]['depth_pred']) ** 2, dim=1).sqrt()
                        epe = epe.view(-1)[valid.view(-1)]
                        one_per = (epe < 0.01*(torch.max(data[view]['depth'])-torch.min(data[view]['depth'])))
                        epe_list.append(epe.mean().item())
                        one_per_list.append(one_per.float().mean().item())

        val_epe = np.round(np.mean(np.array(epe_list)), 4)
        val_one_per = np.round(np.mean(np.array(one_per_list)), 4)
        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        val_ssim = np.round(np.mean(np.array(ssim_list)), 4)

        logging.info(f"Validation Metrics ({self.total_steps}): epe {val_epe}, 1per {val_one_per}, psnr {val_psnr}, ssim {val_ssim}")
        self.logger.write_dict({'val_epe': val_epe, 'val_1per': val_one_per, 'val_ssim':val_ssim, 'val_psnr': val_psnr}, write_step=self.total_steps)
       
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default="config/train.yaml")
    parser.add_argument('--log_path', type=str, default="experiment")
    parser.add_argument('--depth', type=bool, default=True)

    args = parser.parse_args()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load(args.cfg_path)
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.depth = args.depth
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "%s/%s/ckpt" % (args.log_path, cfg.exp_name)
    cfg.record.show_path = "%s/%s/show" % (args.log_path, cfg.exp_name)
    cfg.record.logs_path = "%s/%s/logs" % (args.log_path, cfg.exp_name)
    cfg.record.file_path = "%s/%s/file" % (args.log_path, cfg.exp_name)

    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
