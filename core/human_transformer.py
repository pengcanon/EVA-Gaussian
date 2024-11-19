import torch
import torch.nn as nn
from einops import rearrange
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
import numpy as np
import math

def unproject_depth(depth_map, fxfycxcy, c2w):
    """
    Unproject depth to 3D space (world coordinate).
    """
    B, N, H, W = depth_map.shape
    depth_map = depth_map.reshape(B*N, H, W)
    fxfycxcy = fxfycxcy.reshape(-1, fxfycxcy.shape[-1])
    K = torch.zeros(B*N, 3, 3, device=depth_map.device)
    K[:,0,0] = fxfycxcy[:, 0]
    K[:,1,1] = fxfycxcy[:, 1]
    K[:,0,2] = fxfycxcy[:, 2]
    K[:,1,2] = fxfycxcy[:, 3]
    K[:,2,2] = 1
    c2w = c2w.reshape(B*N, 4, 4)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    y = y.to(depth_map.device).unsqueeze(0).repeat(B*N,1,1)/(H-1)
    x = x.to(depth_map.device).unsqueeze(0).repeat(B*N,1,1)/(W-1)
    xy_map = torch.stack([x, y], axis=-1) * depth_map[..., None]
    xyz_map = torch.cat([xy_map, depth_map[..., None]], axis=-1)
    xyz = xyz_map.view(B*N, -1, 3)

    # get point positions in camera coordinate
    xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), -1, -2))
    xyz_map = xyz.view(B*N, H, W, 3)

    # transform pts from camera to world coordinate
    xyz_homo = torch.ones((B*N, H, W, 4), device=depth_map.device)
    xyz_homo[...,:3] = xyz_map
    xyz_world = torch.bmm(c2w, xyz_homo.reshape(B*N, -1, 4).permute(0, 2, 1)).permute(
                                    0, 2, 1)[..., :3].reshape(B, N*H*W, 3)
    return xyz_world

class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder.
    """
    def __init__(self, input_res, in_channels, patch_size, width, layers, heads, window_size):
        super().__init__()
        self.input_res = input_res
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)

        self.positional_embedding = nn.Parameter(torch.zeros(1, (input_res// patch_size) ** 2, width))
        # self.positional_embedding = nn.Parameter(torch.zeros(1, (input_res) ** 2, width))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.layernorm1 = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, window_size=window_size)
        self.layernorm2 = LayerNorm(width)

    def set_grad_checkpointing(self, set_checkpointing=True):
        self.transformer.set_grad_checkpointing(set_checkpointing)

    def forward(self, x, condition=None):
        _, v, _, h = x.shape[:4]
        x = rearrange(x, 'b v c h w -> (b v) c h w')

        if condition is not None:
            condition = rearrange(condition, 'b v c h w -> (b v) c h w')   
            x = torch.cat([x, condition], dim=1)
        
        x = self.conv(x)
        h = x.shape[2]
        
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.layernorm1(x)
        x = rearrange(x, '(b v) n d -> b (v n) d', v=v)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.layernorm2(x)
        x = rearrange(x, 'b (v n) d -> b v n d', v=v)

        x = rearrange(x, 'b v (h w) d -> b v d h w', h=h)
        return x

class PSUpsamplerBlock(nn.Module):
    """
    Upsampling block.
    """
    def __init__(self, d_model, d_model_out, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor
        self.d_model_out = d_model_out
        self.residual_fc = nn.Linear(d_model, d_model_out * (scale_factor**2))
        self.pixelshuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.residual_fc(x)
        bs, l, c = x.shape
        resolution = int(np.sqrt(l//2))
        x = x.permute(0, 2, 1).reshape(bs, c*2, resolution, resolution)
        x = self.pixelshuffle(x)
        x = x.reshape(bs, self.d_model_out, resolution*self.scale_factor*resolution*self.scale_factor*2)
        x = x.permute(0, 2, 1)
        return x

class GaussianUpsampler(nn.Module):
    """
    Upsampler.
    """
    def __init__(self, width, up_ratio, ch_decay=1, low_channels=64, window_size=None):
        super().__init__()
        self.up_ratio = up_ratio
        self.low_channels = low_channels
        self.window_size = window_size
        self.base_width = width
        self.finalupsample = FinalUpsampler(width, 1, scale_factor=2)
        for res_log2 in range(int(np.log2(up_ratio))):
            _width = width
            # width = max(width // ch_decay, 64)
            heads = int(width / 64)
            # width = heads * 64
            self.add_module(f'upsampler_{res_log2}', PSUpsamplerBlock(_width, width, 2))
            encoder = Transformer(width, 2, heads, window_size=window_size)
            self.add_module(f'attention_{res_log2}', encoder)
        self.out_channels = width
        self.layernorm2 = LayerNorm(width)

    def forward(self, x):
        for res_log2 in range(int(np.log2(self.up_ratio))):
            x = getattr(self, f'upsampler_{res_log2}')(x)
            x = x.permute(1, 0, 2)
            x = getattr(self, f'attention_{res_log2}')(x)
            x = x.permute(1, 0, 2)
        x = self.layernorm2(x)
        x = self.finalupsample(x)
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
    def __init__(self, d_model, n_head, window_size=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", GELU_()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.layernorm2 = LayerNorm(d_model)
        self.window_size = window_size

    def attention(self, x, index):
        attn_mask = None
        if self.window_size is not None:
            x = x.permute(1, 0, 2)
            l = x.shape[1]
            assert l % self.window_size == 0
            if index % 2 == 0:
                x = rearrange(x, 'b (p w) c -> (b p) w c', w=self.window_size)
                x = x.permute(1, 0, 2)
                x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0] 
                x = x.permute(1, 0, 2)
                x = rearrange(x, '(b l) w c -> b (l w) c', l=l//self.window_size, w=self.window_size)
                x = x.permute(1, 0, 2)
            else:
                x = torch.roll(x, shifts=self.window_size//2, dims=1)
                x = rearrange(x, 'b (p w) c -> (b p) w c', w=self.window_size)
                x = x.permute(1, 0, 2)
                x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0] 
                x = x.permute(1, 0, 2)
                x = rearrange(x, '(b l) w c -> b (l w) c', l=l//self.window_size, w=self.window_size)
                x = torch.roll(x, shifts=-self.window_size//2, dims=1)
                x = x.permute(1, 0, 2)
        else:
            x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        return x

    def forward(self, x, index):
        y = self.layernorm1(x)
        y = self.attention(y, index)
        x = x + y
        y = self.layernorm2(x)
        y = self.mlp(y)
        x = x + y
        return x

class Transformer(nn.Module):
    """
    Transformer.
    """
    def __init__(self, width, layers, heads, window_size=None):
        super().__init__()
        self.width = width
        self.layers = layers
        blocks = []
        for _ in range(layers):
            layer = ResAttBlock(width, heads, window_size=window_size) 
            blocks.append(layer)

        self.resblocks = nn.Sequential(*blocks)
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, flag=True):
        self.grad_checkpointing = flag

    def forward(self, x):
        for res_i, module in enumerate(self.resblocks):
            if self.grad_checkpointing:
                x = checkpoint(module, x, res_i)
            else:
                x = module(x, res_i)

        return x
    

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(Upsampler, self).__init__()
        # self.conv = nn.sequential(
        #     *[nn.Conv2d(in_channels, in_channels*scale_factor, 3, 1, 1),
        #       nn.ReLU(),
        #       nn.Conv2d(in_channels*scale_factor, in_channels*scale_factor*scale_factor, 3, 1, 1),])
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.window_attention_1 = WindowAttention(in_channels//(scale_factor**2), in_channels//(scale_factor))
        self.window_attention_2 = WindowAttention(in_channels//(scale_factor), out_channels)
        
    def forward(self, x):
        # x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.window_attention_1(x)
        x = self.window_attention_2(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(WindowAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        attention = self.softmax(x)
        # x = x.view(b, c, h, w)
        x = x * attention.unsqueeze(2)
        x = x.view(b, -1, h, w)
        return x
    
class FinalUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(FinalUpsampler, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.conv1 = nn.Sequential(*[nn.Conv2d(in_channels//(scale_factor**2), in_channels//(scale_factor),3,stride=1,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels//(scale_factor), in_channels//(scale_factor),3,stride=1,padding=1),
                                     ])
        self.conv2 = nn.Sequential(*[nn.Conv2d(in_channels//(scale_factor**3), in_channels//(scale_factor**2),3,stride=1,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels//(scale_factor**2), in_channels//(scale_factor**2),3,stride=1,padding=1),
                                     ])
        self.conv3 = nn.Sequential(*[nn.Conv2d(in_channels//(scale_factor**4), in_channels//(scale_factor**2),3,stride=1,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels//(scale_factor**2), in_channels//(scale_factor**2),3,stride=1,padding=1),
                                     ])
        self.conv4 = nn.Sequential(*[nn.Conv2d(in_channels//(scale_factor**4), in_channels//(scale_factor**4),3,stride=1,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels//(scale_factor**4), out_channels,3,stride=1,padding=1),
                                     ])
        
    def forward(self, x):
        # x = self.conv(x)
        bs,f,c=x.shape
        h=int(math.sqrt(f/2))
        x = rearrange(x, "b (v h w) c  -> (b v) c h w", b=bs, v=2, h=h, w=h)
        x = self.pixel_shuffle(x)
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = self.conv4(x)
        return x