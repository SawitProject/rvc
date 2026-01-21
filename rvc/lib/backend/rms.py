import torch
import librosa

import torch.nn as nn

class RMSEnergyExtractor(nn.Module):
    def __init__(self, frame_length=2048, hop_length=512, center=True, pad_mode = "reflect"):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode

    def forward(self, x):
        assert x.ndim == 2
        assert x.shape[0] == 1

        if str(x.device).startswith("ocl"): x = x.contiguous()

        rms = torch.from_numpy(
            librosa.feature.rms(
                y=x.squeeze(0).cpu().numpy(), 
                frame_length=self.frame_length, 
                hop_length=self.hop_length, 
                center=self.center, 
                pad_mode=self.pad_mode
            )
        )

        return rms.squeeze(-2).to(x.device) if not str(x.device).startswith("ocl") else rms.contiguous().squeeze(-2).to(x.device)