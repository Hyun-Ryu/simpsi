import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Equalizer(nn.Module):
    def __init__(self, configs, args):
        super(Equalizer, self).__init__()

        # select prior
        if args.prior == 'mag' or args.prior == 'slc':
            c_in = 2
        elif args.prior == 'self':
            c_in = 4
        else:
            c_in = 1

        # select equalizer architecture
        self.equalizer = args.equalizer
        if args.equalizer == 'conv':
            self.eq = nn.Conv1d(
                in_channels = c_in,
                out_channels = 1,
                kernel_size = args.eq_kernel_size,
                padding = args.eq_kernel_size//2,
                bias = False)
        elif args.equalizer == 'transformer':
            num_time_steps = 2500
            num_freq_steps = num_time_steps//2 + 1
            encoder_layers = TransformerEncoderLayer(
                d_model = num_freq_steps,
                nhead = 1,
                dim_feedforward = 2*num_freq_steps,
                batch_first = True)
            self.eq = TransformerEncoder(encoder_layers, 2)

    def forward(self, I_f):
        if self.equalizer == 'conv':
            return F.sigmoid(self.eq(I_f))
        elif self.equalizer == 'transformer':
            return F.sigmoid(torch.mean(self.eq(I_f), dim=1, keepdim=True))
