
import torch
from torch import nn
from core.human_transformer import TransformerEncoder, GaussianUpsampler
from core.extractor import UnetExtractor
from lib.attention_unet import AttenUNet
from lib.gs_parm_network import GSRegresser_feature, feature_refiner
from lib.loss import sequence_depth_loss
from lib.utils import depth2pc
from torch.cuda.amp import autocast as autocast
from einops import rearrange


class EVANet(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render

        self.Unet = AttenUNet(3, 1)
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser_feature(self.cfg, rgb_dim=3, depth_dim=1)
            self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=[32, 48, 96])
            self.feature_refiner = feature_refiner()

        self.sigmoid = nn.Sigmoid()
        self.regularize = nn.Sigmoid()

    def forward(self, data, is_train=True):
        bs = data['lmain']['img'].shape[0]

        image = torch.stack([data['lmain']['img'], data['rmain']['img']], dim=1)
        depth = torch.stack([data['lmain']['depth'], data['rmain']['depth']], dim=1)
        extrinsics = torch.stack([data['lmain']['extr'], data['rmain']['extr']], dim=1)
        intrinsics = torch.stack([data['lmain']['intr'], data['rmain']['intr']], dim=1)

        bs,v,c,h,w = image.shape
        img_feat = None

        if is_train:
            features = self.Unet(image)
            predictions = rearrange(features, "(b v) c h w  -> b v c h w", b=bs, v=v)
            pred_lmain, pred_rmain = predictions[:,0], predictions[:,1]

            depth_loss, metrics = sequence_depth_loss([predictions[:,:,:1]], depth)

            data['lmain']['depth_pred'] = pred_lmain[:,:1]
            data['rmain']['depth_pred'] = pred_rmain[:,:1]

            for view in ['lmain', 'rmain']:
                data[view]['xyz'] = depth2pc(data[view]['depth_pred'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
                valid = data[view]['depth'] != 0.0
                data[view]['pts_valid'] = valid.view(bs, -1)

            if not self.with_gs_render:
                return data, depth_loss, metrics
            
            image_encoder = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
            img_feat = self.img_encoder(image_encoder)
            data = self.depth2gsparms(image_encoder, img_feat, data, bs)

            return data, depth_loss, metrics

        else:
            features = self.Unet(image)
            predictions = rearrange(features, "(b v) c h w  -> b v c h w", b=bs, v=v)
            pred_lmain, pred_rmain = predictions[:,0], predictions[:,1]
            
            depth_loss, metrics = sequence_depth_loss([predictions[:,:,:1]], depth)

            data['lmain']['depth_pred'] = pred_lmain[:,:1]
            data['rmain']['depth_pred'] = pred_rmain[:,:1]

            for view in ['lmain', 'rmain']:
                data[view]['xyz'] = depth2pc(data[view]['depth_pred'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
                valid = data[view]['depth'] != 0.0
                data[view]['pts_valid'] = valid.view(bs, -1)

            if not self.with_gs_render:
                return data, depth_loss, metrics
            
            image_encoder = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
            img_feat = self.img_encoder(image_encoder)

            data = self.depth2gsparms(image_encoder, img_feat, data, bs)

            return data, depth_loss, metrics

    def depth2gsparms(self, lr_img, lr_img_feat, data, bs):
        lr_depth = torch.concat([data['lmain']['depth_pred'], data['rmain']['depth_pred']], dim=0)
        
        rot_maps, scale_maps, opacity_maps, feature_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)

        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])
        data['lmain']['features'], data['rmain']['features'] = torch.split(feature_maps, [bs, bs])

        return data

