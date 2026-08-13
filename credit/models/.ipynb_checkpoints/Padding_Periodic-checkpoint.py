import torch
import logging
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from credit.models.base_model import BaseModel
from credit.postblock import PostBlock

class PeriodicLon_RotRefLat(torch.nn.Module):
    def __init__(self, N_S=40, E_W=80):
        super(PeriodicLon_RotRefLat, self).__init__()
        # Compute padding based on the kernel size (integer division for "same" padding)
        self.paddingNS = N_S
        self.paddingEW = E_W

    def forward(self, x):
        # x shape is [batch, variable, latitude, longitude]
        #roll and reflect latitude 

        if self.paddingNS>0:
            xroll = torch.roll(x, shifts=x.shape[3] // 2, dims=3)  # 180-degree shift (half the longitude size)
            xroll_flip_top = torch.flip(xroll[:,:,:self.paddingNS,:],(2,))
            xroll_flip_bot = torch.flip(xroll[:,:,-self.paddingNS:,:],(2,))
            x = torch.cat([xroll_flip_top, x, xroll_flip_bot], dim=-2)

        # Circular lon
        x = F.pad(x, (self.paddingEW, self.paddingEW, 0, 0), mode='circular')  # Pad last dimension (longitude) by self.padding
        return x