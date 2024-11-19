import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from collections import OrderedDict
from typing import Tuple, Literal
from functools import partial
import torchvision
import functools

import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 16,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x


class LayerNorm(nn.LayerNorm):
    """
    To support fp16.
    """
    def forward(self, x):
        type_ = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(type_)


class GELU_(nn.Module):
    """
    Fast gelu implementation.
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResAttBlock(nn.Module):
    """
    Attention block.
    """
    def __init__(self, d_model, n_head, window_size=None, dropout=0.4):
        super().__init__()
        self.layernorm1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", GELU_()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.layernorm2 = LayerNorm(d_model)
        self.window_size = window_size

        self.attn_1 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attn_2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
    
    def attention(self, x, y, index):
        attn_mask = None
        if self.window_size is not None:
            l = x.shape[2]
            assert l % self.window_size == 0
            if index % 2 == 0:
                x = rearrange(x, 'b v (p w) c -> (b p) v w c', w=self.window_size)  
                y = rearrange(y, 'b v (p w) c -> (b p) v w c', w=self.window_size)          
                xv1, xv2 = x[:,0], x[:,1]
                yv1, yv2 = y[:,0], y[:,1]
                yv1 = yv1.permute(1, 0, 2)
                yv2 = yv2.permute(1, 0, 2)
                xv1 = xv1.permute(1, 0, 2)
                xv2 = xv2.permute(1, 0, 2)
                v1 = self.attn_1(yv1, yv2, xv1, need_weights=False, attn_mask=attn_mask)[0] 
                v1 = v1.permute(1, 0, 2)             

                v2 = self.attn_2(yv2, yv1, xv2, need_weights=False, attn_mask=attn_mask)[0] 
                v2 = v2.permute(1, 0, 2)
                x = torch.cat((v1.unsqueeze(dim=1), v2.unsqueeze(dim=1)), dim=1)
                x = rearrange(x, '(b l) v w c -> b v (l w) c', l=l//self.window_size, w=self.window_size)
            else:
                x = torch.roll(x, shifts=self.window_size//2, dims=2)
                x = rearrange(x, 'b v (p w) c -> (b p) v w c', w=self.window_size)  
                y = rearrange(y, 'b v (p w) c -> (b p) v w c', w=self.window_size)          
                xv1, xv2 = x[:,0], x[:,1]
                yv1, yv2 = y[:,0], y[:,1]
                yv1 = yv1.permute(1, 0, 2)
                yv2 = yv2.permute(1, 0, 2)
                xv1 = xv1.permute(1, 0, 2)
                xv2 = xv2.permute(1, 0, 2)
                v1 = self.attn_1(yv1, yv2, xv1, need_weights=False, attn_mask=attn_mask)[0] 
                v1 = v1.permute(1, 0, 2)             

                v2 = self.attn_2(yv2, yv1, xv2, need_weights=False, attn_mask=attn_mask)[0] 
                v2 = v2.permute(1, 0, 2)
                x = torch.cat((v1.unsqueeze(dim=1), v2.unsqueeze(dim=1)), dim=1)
                x = rearrange(x, '(b l) v w c -> b v (l w) c', l=l//self.window_size, w=self.window_size)
                x = torch.roll(x, shifts=-self.window_size//2, dims=2)
        else:            
            xv1, xv2 = x[:,0], x[:,1]
            yv1, yv2 = y[:,0], y[:,1]
            yv1 = yv1.permute(1, 0, 2)
            yv2 = yv2.permute(1, 0, 2)
            xv1 = xv1.permute(1, 0, 2)
            xv2 = xv2.permute(1, 0, 2)
            v1 = self.attn_1(yv1, yv2, xv1, need_weights=False, attn_mask=attn_mask)[0] 
            v1 = v1.permute(1, 0, 2)             

            v2 = self.attn_2(yv2, yv1, xv2, need_weights=False, attn_mask=attn_mask)[0] 
            v2 = v2.permute(1, 0, 2)
            x = torch.cat((v1.unsqueeze(dim=1), v2.unsqueeze(dim=1)), dim=1)

        return x
        
    def forward(self, x, index=0):
        y = self.layernorm1(x)
        y = self.attention(x, y, index)
        x = x + y
        y = self.layernorm2(x)
        y = self.mlp(y)
        x = x + y
        return x


class CVA(nn.Module):
    def __init__(self, d_model, v, window_size=None, res=1024):
        super().__init__()
        self.v = v
        self.cross_attn_1 = ResAttBlock(d_model, n_head=d_model//32, window_size=window_size)
        self.cross_attn_2 = ResAttBlock(d_model, n_head=d_model//32, window_size=window_size)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.d_model = d_model
        
        self.positional_embedding = nn.Parameter(torch.zeros(1, v, (res ** 2), d_model))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        _, c, h, w = x.shape
        x = rearrange(x, '(b v) c h w -> b v (h w) c', c=c, h=h, w=w, v=self.v)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.layernorm1(x)
        x = self.cross_attn_1(x, index=0)
        x = self.cross_attn_2(x, index=1)
        x = self.layernorm2(x)
        x = rearrange(x, 'b v (h w) c -> (b v) c h w', c=c, h=h, w=w, v=self.v)

        return x


class AttenUNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=9, v=2):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.input = nn.Conv2d(in_channel, 16, 1, 1, 0)
        # left
        self.left_conv_1 = DoubleConv(16, 16, groups=16)
        self.down_1 = nn.AvgPool2d(2, 2)

        self.left_conv_2 = DoubleConv(16, 32, groups=16)
        self.down_2 = nn.AvgPool2d(2, 2)

        self.left_conv_3 = DoubleConv(32, 64)
        self.atten_down_3 = CVA(d_model=64, v=v, window_size=32, res=256)
        self.down_3 = nn.AvgPool2d(2, 2)

        self.left_conv_4 = DoubleConv(64, 128)
        self.atten_down_4 = CVA(d_model=128, v=v, window_size=64, res=128)
        self.down_4 = nn.AvgPool2d(2, 2)

        # center
        self.center_conv = DoubleConv(128, 256)
        self.center_atten = CVA(d_model=256, v=v, window_size=64, res=64)
        # right

        self.up_1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_1 = DoubleConv(256, 128)
        self.atten_up_1 = CVA(d_model=128, v=v, window_size=64, res=128)

        self.up_2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_2 = DoubleConv(128, 64)
        self.atten_up_2 = CVA(d_model=64, v=v, window_size=32, res=256)

        self.up_3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.right_conv_3 = DoubleConv(64, 32)

        self.up_4 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.right_conv_4 = DoubleConv(32, 16, groups=16)

        # output
        self.norm_out = nn.GroupNorm(num_channels=16, num_groups=16, eps=1e-5)
        self.output = nn.Conv2d(16, out_channel, 1, 1, 0)

    def forward(self, x):
        _, v, _, h = x.shape[:4]
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        # left
        x = self.input(x)

        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3 = self.atten_down_3(x3)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4 = self.atten_down_4(x4)
        x4_down = self.down_4(x4)

        # center
        x5 = self.center_conv(x4_down)
        x5 = self.center_atten(x5)

        # right
        x6_up = self.up_1(x5)
        temp = torch.cat((x6_up, x4), dim=1)
        x7 = self.right_conv_1(temp)
        x7 = self.atten_up_1(x7)

        x7_up = self.up_2(x7)
        temp = torch.cat((x7_up, x3), dim=1)
        x8 = self.right_conv_2(temp)
        x8 = self.atten_up_2(x8)

        x8_up = self.up_3(x8)
        temp = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        temp = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(temp)

        # output
        x9 = self.norm_out(x9)
        output = self.sigmoid(self.output(F.silu(x9)))

        return output

