import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Tools import generate_continuous_mask, generate_binomial_mask


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn=True, dilation=1, final=False):
        super().__init__()
        self.resconnector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels) if bn else None
        self.bn2 = nn.BatchNorm1d(out_channels) if bn else None
        self.gelu = nn.GELU()
    
    def forward(self, x):
        residual = x if self.resconnector is None else self.resconnector(x)
        out = self.conv1(x)
        out = out if self.bn1 is None else self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = out if self.bn2 is None else self.bn2(out)
        out = out + residual
        out = self.gelu(out)
        return out

class DilatedConvEncoder(nn.Module): # cnn
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ResBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
    
class TSEncoder(nn.Module): # TrEncoder
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x

class TrEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', channels=23, individual=True):
        super().__init__()
        self.output_dims = output_dims
        self.channels = channels
        self.individual = individual
        if self.individual:
            self.net = nn.ModuleList([
                TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, mask_mode=mask_mode)
                for _ in range(channels)
            ])
        else:
            self.net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
    
    def forward(self, x, mask=None):
        if self.individual:
            x = x.reshape(x.size(0), x.size(1), x.size(2), 1)
            ls = []
            for i, net in enumerate(self.net):
                ls.append(net(x[:, :, i].clone(), mask))
            x = torch.stack(ls, dim=-1)
            x = x.reshape(x.size(0), x.size(1), -1)
        else:
            x = self.net(x, mask)
            
        return x