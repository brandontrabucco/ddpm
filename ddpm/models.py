import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Residual(nn.Module):
    
    def __init__(self, dim: int, 
                 dim_feedforward: int = 32, 
                 dropout: float = 0.1):
        
        super(Residual, self).__init__()
        
        self.norm = nn.GroupNorm(1, dim)
        
        self.feedforward = nn.Sequential(
            nn.Conv2d(dim + 24, dim_feedforward, 3, padding=1),
            nn.GELU(), 
            nn.Dropout2d(p=dropout),
            nn.Conv2d(dim_feedforward, dim, 3, padding=1),
            nn.Dropout2d(p=dropout),
        )
        
    def forward(self, x, timestep):
        
        pos = positional_encoding(timestep.float(), -11, 12)
        pos = pos.view(x.shape[0], 24, 1, 1)
        pos = pos.expand(x.shape[0], 24, *x.shape[2:])
        
        return x + self.feedforward(
            torch.cat((self.norm(x), pos), dim=1))
    

class ResidualBlock(nn.Module):
    
    def __init__(self, dim: int, **kwargs):
        
        super(ResidualBlock, self).__init__()
        
        self.layer1 = Residual(dim, **kwargs)
        self.layer2 = Residual(dim, **kwargs)
        self.layer3 = Residual(dim, **kwargs)
        
    def forward(self, x, timestep):
        
        x = self.layer1(x, timestep)
        x = self.layer2(x, timestep)
        return self.layer3(x, timestep)
    

class UNetBlock(nn.Module):
    
    def __init__(self, dim: int, 
                 dim_feedforward: int = 32, 
                 dropout: float = 0.1, 
                 inner_block: nn.Module = None):
        
        super(UNetBlock, self).__init__()
        
        self.downsample = nn.PixelUnshuffle(2)
        self.upsample = nn.PixelShuffle(2)
        
        self.downsample_module = ResidualBlock(
            dim, dim_feedforward=
            dim_feedforward, dropout=dropout)
        
        self.upsample_module = ResidualBlock(
            dim, dim_feedforward=
            dim_feedforward, dropout=dropout)
        
        self.inner_block = inner_block
        
    def forward(self, x, timestep):
        
        x = self.downsample_module(x, timestep)
        x0, x1 = torch.chunk(x, 2, dim=1)
        
        x0 = self.downsample(x0)
        
        if self.inner_block is not None:
            x0 = self.inner_block(x0, timestep)
            
        x0 = self.upsample(x0)
        
        x = torch.cat((x0, x1), dim=1)
        return self.upsample_module(x, timestep)


class DiffusionModel(nn.Module):
    
    def __init__(self, dim: int = 192, 
                 dim_feedforward: int = 384, 
                 dropout: float = 0.1,
                 num_resolutions: int = 3):
        
        super(DiffusionModel, self).__init__()
        num_resolutions -= 1
        
        self.conv1 = nn.Conv2d(3, dim, 1)
        self.conv2 = nn.Conv2d(dim, 3, 1)
        
        self.unet = ResidualBlock(
            dim * 2 ** num_resolutions, 
            dim_feedforward=
            dim_feedforward * 2 ** num_resolutions,
            dropout=dropout
        )
        
        for layer in reversed(
                range(num_resolutions)):
            
            self.unet = UNetBlock(
                dim * 2 ** layer, 
                dim_feedforward=
                dim_feedforward * 2 ** layer,
                dropout=dropout,
                inner_block=self.unet
            )
        
    def forward(self, x, timestep):
        
        return self.conv2(
            self.unet(self.conv1(x), timestep))