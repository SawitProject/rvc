import os
import sys
import torch
import torch.nn.functional as F

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from modules.commons import init_weights
from modules.residuals import ResBlock, LRELU_SLOPE

class HiFiGANGenerator(torch.nn.Module):
    def __init__(self, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(HiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups_and_resblocks = torch.nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups_and_resblocks.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.ups_and_resblocks.append(ResBlock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups_and_resblocks.apply(init_weights)
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        def forward(self, x, g = None):
            x = self.conv_pre(x)
            if g is not None: x = x + self.cond(g)
            
            resblock_idx = 0

            for _ in range(self.num_upsamples):
                x = self.ups_and_resblocks[resblock_idx](F.leaky_relu(x, LRELU_SLOPE))
                resblock_idx += 1
                xs = 0

                for _ in range(self.num_kernels):
                    xs += self.ups_and_resblocks[resblock_idx](x)
                    resblock_idx += 1

                x = xs / self.num_kernels

            return torch.tanh(self.conv_post(F.leaky_relu(x)))

    def __prepare_scriptable__(self):
        for l in self.ups_and_resblocks:
            for hook in l._forward_pre_hooks.values():
                if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"): torch.nn.utils.remove_weight_norm(l)

        return self
    
    def remove_weight_norm(self):
        for l in self.ups_and_resblocks:
            remove_weight_norm(l)