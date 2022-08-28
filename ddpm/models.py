import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class ResidualLayer(nn.Module):
    
    def __init__(self, dim: int, 
                 embed_dim: int,
                 inner_dim: int = 384, 
                 groups: int = 1,
                 dropout: float = 0.0):
        
        super(ResidualLayer, self).__init__()
        
        self.norm = nn.GroupNorm(groups, dim)
        self.embed_to_param = nn.Linear(embed_dim, 2 * dim)
        
        self.feedforward = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 3, padding=1),
            nn.GELU(), 
            nn.Dropout2d(p=dropout),
            nn.Conv2d(inner_dim, dim, 3, padding=1),
            nn.Dropout2d(p=dropout))
        
    def forward(self, x, embed):

        scale, shift = self.embed_to_param(embed).chunk(2, dim=1)
        scale = scale.view(*x.shape[:2], 1, 1)
        shift = shift.view(*x.shape[:2], 1, 1)

        return x + self.feedforward(
            self.norm(x)) * (scale + 1) + shift
    

class ResidualBlock(nn.Module):
    
    def __init__(self, *args, **kwargs):
        
        super(ResidualBlock, self).__init__()
        
        self.layer1 = ResidualLayer(*args, **kwargs)
        self.layer2 = ResidualLayer(*args, **kwargs)
        self.layer3 = ResidualLayer(*args, **kwargs)
        
    def forward(self, x, embed):
        
        x = self.layer1(x, embed)
        x = self.layer2(x, embed)
        return self.layer3(x, embed)
    

class UNetBlock(nn.Module):
    
    def __init__(self, *args, inner_module: 
                 nn.Module = None, **kwargs):
        
        super(UNetBlock, self).__init__()
        
        self.downsample = nn.PixelUnshuffle(2)
        self.upsample = nn.PixelShuffle(2)
        
        self.downsample_module = \
            ResidualBlock(*args, **kwargs)
        
        self.upsample_module = \
            ResidualBlock(*args, **kwargs)
        
        self.inner_module = inner_module
        
    def forward(self, x, embed):
        
        x = self.downsample_module(x, embed)
        x0, x1 = torch.chunk(x, 2, dim=1)

        x0 = self.downsample(x0)

        if self.inner_module is not None:
            x0 = self.inner_module(x0, embed)

        x0 = self.upsample(x0)

        x = torch.cat((x0, x1), dim=1)
        return self.upsample_module(x, embed)


def positional_encoding(coords, start, num_octaves):

    coords_shape = coords.shape

    octaves = torch.arange(start, start + num_octaves)
    octaves = octaves.float().to(coords.device)

    multipliers = (2 ** octaves) * 3.1415927410125732

    coords = coords.unsqueeze(-1)
    while len(multipliers.shape) < len(coords.shape):
        multipliers = multipliers.unsqueeze(0)

    scaled_coords = coords * multipliers

    return torch.cat((torch.sin(scaled_coords),
                      torch.cos(scaled_coords)), dim=-1)


class PositionalEncoding(nn.Module):

    def __init__(self, start: int, num_octaves: int):
        
        super(PositionalEncoding, self).__init__()

        self.start = start
        self.num_octaves = num_octaves

    def forward(self, x):

        return positional_encoding(
            x, self.start, self.num_octaves)


class DiffusionModel(nn.Module):
    
    def __init__(self, base_dim: int = 192, 
                 base_inner_dim: int = 384, 
                 embed_dim: int = 384, 
                 embed_start_octave: int = -1, 
                 embed_num_octaves: int = 12, 
                 dropout: float = 0.0,
                 num_resolutions: int = 3):
        
        super(DiffusionModel, self).__init__()

        self.timestep_embed = nn.Sequential(
            PositionalEncoding(embed_start_octave, embed_num_octaves),
            nn.Linear(embed_num_octaves * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU())
        
        self.conv1 = nn.Conv2d(3, base_dim, 1)
        self.conv2 = nn.Conv2d(base_dim, 3, 1)

        dim_mults = [2 ** n for n in range(
            num_resolutions - 1, -1, -1)]
        
        self.unet = ResidualBlock(
            base_dim * dim_mults[0], embed_dim,
            inner_dim=base_inner_dim * dim_mults[0],
            groups=dim_mults[0],
            dropout=dropout)
        
        for multx in dim_mults[1:]:
            
            self.unet = UNetBlock(
                base_dim * multx, embed_dim,
                inner_dim=base_inner_dim * multx,
                groups=multx,
                dropout=dropout,
                inner_module=self.unet)
        
    def forward(self, x, timestep):

        embed = self.timestep_embed(timestep)
        return self.conv2(self.unet(self.conv1(x), embed))