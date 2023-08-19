import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Equalizer(nn.Module):
    def __init__(self, args):
        super(Equalizer, self).__init__()
        # select preservation map
        if args.prior == 'mag':
            c_in = 1
        elif args.prior == 'slc' or args.prior == 'self':
            c_in = 2

        # select equalizer
        self.equalizer = args.equalizer
        if args.equalizer == 'transformer':
            num_freq_steps = 128
            encoder_layers = TransformerEncoderLayer(
                d_model = num_freq_steps,
                nhead = 1,
                dim_feedforward = 2*num_freq_steps,
                batch_first = True)
            self.eq = TransformerEncoder(encoder_layers, 2)
        elif args.equalizer == 'conv':
            self.eq = nn.Conv1d(
                in_channels = c_in,
                out_channels = 1,
                kernel_size = args.eq_kernel_size,
                padding = args.eq_kernel_size//2,
                bias = False)

    def forward(self, I_f):
        # I_f: (b, 2, 128)
        # return: (b, 1, 128)
        if self.equalizer == 'conv':
            return F.sigmoid(self.eq(I_f.float()))
        elif self.equalizer == 'transformer':
            return F.sigmoid(torch.mean(self.eq(I_f), dim=1, keepdim=True))


class ResidualUnit(nn.Module):
    def __init__(self, dim):
        super(ResidualUnit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1,5), stride=1, padding=(0,2), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=dim),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1,5), stride=1, padding=(0,2), bias=True),
            nn.BatchNorm2d(num_features=dim),
        )
    
    def forward(self, x):
        return x + self.conv(x)
    
class ResidualStack(nn.Module):
    def __init__(self, in_dim, hid_dim, pool=True):
        super(ResidualStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1,1), stride=1, padding=(0,0), bias=True),
            nn.BatchNorm2d(num_features=hid_dim),
        )
        self.residualunit1 = ResidualUnit(dim=hid_dim)
        self.residualunit2 = ResidualUnit(dim=hid_dim)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,2))
        self.pool = pool
    
    def forward(self, x):
        out = self.conv(x)
        out = self.residualunit1(out)
        out = self.residualunit2(out)
        if self.pool:
            out = self.maxpool(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, dim_input=2, dim_hidden=32, dim_output=2, pool=True):
        super(ResNet1D, self).__init__()
        self.residualstack = nn.Sequential(
            ResidualStack(in_dim=dim_input, hid_dim=dim_hidden, pool=pool),
            ResidualStack(in_dim=dim_hidden, hid_dim=dim_output, pool=pool),
        )
    
    def forward(self, x):
        # x: (b, 2, 1, 128)
        out = self.residualstack(x)
        # out: (b, C, 1, 32)
        out = out.squeeze(2)
        # out: (b, C, 32)
        return out