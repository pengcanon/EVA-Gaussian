
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from core.extractor import UnetExtractor, ResidualBlock
import functools
from functools import partial
from collections import OrderedDict
import tqdm

class GSRegresser_feature(nn.Module):
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.rgb_dims = [32, 48, 96]
        self.depth_dims = [32, 48, 96]
        self.decoder_dims = [48, 64, 96]
        self.head_dim = 16
        self.feat_dim = 32
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0]+rgb_dim+1, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Sigmoid()
            # nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.feature_head = nn.Sequential(
            ResidualBlock(self.head_dim+4, self.head_dim, norm_fn=norm_fn),
            ResidualBlock(self.head_dim, self.head_dim, norm_fn=norm_fn),
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, self.feat_dim, kernel_size=1),
        )

    def forward(self, img, depth, img_feat):
        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)

        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        scale_out = 0.01*self.scale_head(out)

        opacity_out = self.opacity_head(out)

        feat_out = self.feature_head(torch.cat((out, img, depth), dim=1))

        return rot_out, scale_out, opacity_out, feat_out


class feature_refiner(nn.Module):
    def __init__(self, feat_dim=32, norm_fn='group'):
        super().__init__()
        
        self.rgb_dims = [32, 48, 96]
        self.decoder_dims = [32, 64, 96]

        self.feat_extractor_1 = nn.Sequential(*[nn.Conv2d(3+feat_dim, 32, kernel_size=5, stride=1, padding=2),
                nn.GroupNorm(num_groups=8, num_channels=32),
                nn.ReLU(inplace=True),
                ResidualBlock(32, self.rgb_dims[0], stride=2, norm_fn=norm_fn),
                ResidualBlock(self.rgb_dims[0], self.rgb_dims[0], norm_fn=norm_fn),])
        
        self.feat_extractor_2 = nn.Sequential(*[
                ResidualBlock(self.rgb_dims[0], self.rgb_dims[1], stride=2, norm_fn=norm_fn),
                ResidualBlock(self.rgb_dims[1], self.rgb_dims[1], norm_fn=norm_fn),])
        
        self.feat_extractor_3 = nn.Sequential(*[
                ResidualBlock(self.rgb_dims[1], self.rgb_dims[2], stride=2, norm_fn=norm_fn),
                ResidualBlock(self.rgb_dims[2], self.rgb_dims[2], norm_fn=norm_fn),])
        
        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.decoder_dims[2]+self.rgb_dims[1], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.decoder_dims[1]+self.rgb_dims[0], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )

        self.up3 = nn.ConvTranspose2d(self.decoder_dims[2], self.decoder_dims[2], 2, 2)
        self.up2 = nn.ConvTranspose2d(self.decoder_dims[1], self.decoder_dims[1], 2, 2)
        self.up1 = nn.ConvTranspose2d(self.decoder_dims[0], self.decoder_dims[0], 2, 2)
        
        self.out_conv = nn.Sequential(
            ResidualBlock(self.decoder_dims[0]+3+feat_dim, self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn),
            nn.Conv2d(self.decoder_dims[0], self.decoder_dims[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.decoder_dims[0], 3, kernel_size=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feat):
        rgb_feat = torch.concat([x,feat],dim=1)

        feat1 = self.feat_extractor_1(rgb_feat)
        feat2 = self.feat_extractor_2(feat1)
        feat3 = self.feat_extractor_3(feat2)

        up3 = self.decoder3(feat3)
        up3 = self.up3(up3)
        up2 = self.decoder2(torch.cat([up3,feat2], dim=1))
        up2 = self.up2(up2)
        up1 = self.decoder1(torch.cat([up2,feat1], dim=1))

        up1 = self.up1(up1)
        out = self.sigmoid(self.out_conv(torch.cat([up1,rgb_feat], dim=1)))

        return [out]

