import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import medfilt
from torch import nn
from einops.layers.torch import Rearrange

sys.path.append(os.getcwd())

# Constants
SAMPLE_RATE, WINDOW_LENGTH, N_CLASS = 16000, 1024, 360

# ==================== Utils Module ====================
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None: 
        layer.bias.data.fill_(0.0)

def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)

class BiGRU(nn.Module):
    def __init__(self, patch_size, channels, depth):
        super(BiGRU, self).__init__()
        patch_width, patch_height = patch_size
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p1) (h p2) -> b (w h) (p1 p2 c)', 
                     p1=patch_width, p2=patch_height)
        )

        self.gru = nn.GRU(
            patch_dim, 
            patch_dim // 2, 
            num_layers=depth, 
            batch_first=True, 
            bidirectional=True
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        try:
            return self.gru(x)[0]
        except:
            torch.backends.cudnn.enabled = False
            return self.gru(x)[0]

class ResConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, (3, 3), padding=(1, 1), bias=False)
        self.is_shortcut = False

        if in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, (1, 1))
            self.is_shortcut = True

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)
        if self.is_shortcut: 
            init_layer(self.shortcut)

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        if self.is_shortcut: 
            return self.shortcut(x) + out
        else: 
            return out + x

# ==================== Spectrogram Module ====================
class Spectrogram(nn.Module):
    def __init__(self, hop_length, win_length, n_fft=None, clamp=1e-10):
        super(Spectrogram, self).__init__()
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, audio, center=True):
        bs, c, segment_samples = audio.shape
        audio = audio.reshape(bs * c, segment_samples)

        if str(audio.device).startswith(("ocl", "privateuseone")):
            if not hasattr(self, "stft"): 
                from main.library.backends.utils import STFT
                self.stft = STFT(
                    filter_length=self.n_fft, 
                    hop_length=self.hop_length, 
                    win_length=self.win_length
                ).to(audio.device)
            magnitude = self.stft.transform(audio, 1e-9)
        else:
            fft = torch.stft(
                audio, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length, 
                window=self.window, 
                center=center, 
                pad_mode="reflect", 
                return_complex=True
            )
            magnitude = (fft.real.pow(2) + fft.imag.pow(2)).sqrt()

        mag = magnitude.transpose(1, 2).clamp(self.clamp, np.inf)
        mag = mag.reshape(bs, c, mag.shape[1], mag.shape[2])
        return mag

# ==================== Encoder Module ====================
class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, kernel_size):
        super(ResEncoderBlock, self).__init__()
        self.conv = nn.ModuleList([ResConvBlock(in_channels, out_channels)])
        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))
        self.pool = nn.MaxPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, x):
        for each_layer in self.conv:
            x = each_layer(x)
        if self.pool is not None: 
            return x, self.pool(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, n_blocks):
        super(Encoder, self).__init__()
        self.en_blocks = nn.ModuleList([
            ResEncoderBlock(in_channels, 32, n_blocks, (1, 2)), 
            ResEncoderBlock(32, 64, n_blocks, (1, 2)), 
            ResEncoderBlock(64, 128, n_blocks, (1, 2)), 
            ResEncoderBlock(128, 256, n_blocks, (1, 2)), 
            ResEncoderBlock(256, 384, n_blocks, (1, 2)), 
            ResEncoderBlock(384, 384, n_blocks, (1, 2))
        ])

    def forward(self, x):
        concat_tensors = []
        for layer in self.en_blocks:
            _, x = layer(x)
            concat_tensors.append(_)
        return x, concat_tensors

# ==================== Decoder Module ====================
class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride):
        super(ResDecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, stride, stride, (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv = nn.ModuleList([ResConvBlock(out_channels * 2, out_channels)])
        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, x, concat):
        x = self.conv1(F.relu_(self.bn1(x)))
        x = torch.cat((x, concat), dim=1)
        for each_layer in self.conv:
            x = each_layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n_blocks):
        super(Decoder, self).__init__()
        self.de_blocks = nn.ModuleList([
            ResDecoderBlock(384, 384, n_blocks, (1, 2)), 
            ResDecoderBlock(384, 384, n_blocks, (1, 2)), 
            ResDecoderBlock(384, 256, n_blocks, (1, 2)), 
            ResDecoderBlock(256, 128, n_blocks, (1, 2)), 
            ResDecoderBlock(128, 64, n_blocks, (1, 2)), 
            ResDecoderBlock(64, 32, n_blocks, (1, 2))
        ])

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.de_blocks):
            x = layer(x, concat_tensors[-1 - i])
        return x

class PE_Decoder(nn.Module):
    def __init__(self, n_blocks, seq_layers=1, window_length=1024, n_class=360):
        super(PE_Decoder, self).__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None)
        self.after_conv2 = nn.Conv2d(32, 1, (1, 1))
        self.fc = nn.Sequential(
            BiGRU((1, window_length // 2), 1, seq_layers), 
            nn.Linear(window_length // 2, n_class), 
            nn.Sigmoid()
        )
        init_layer(self.after_conv2)

    def forward(self, x, concat_tensors):
        return self.fc(self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))).squeeze(1)

class SVS_Decoder(nn.Module):
    def __init__(self, in_channels, n_blocks):
        super(SVS_Decoder, self).__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None)
        self.after_conv2 = nn.Conv2d(32, in_channels * 4, (1, 1))
        self.init_weights()

    def init_weights(self):
        init_layer(self.after_conv2)

    def forward(self, x, concat_tensors):
        return self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))

# ==================== Main Model ====================
class LatentBlocks(nn.Module):
    def __init__(self, n_blocks, latent_layers):
        super(LatentBlocks, self).__init__()
        self.latent_blocks = nn.ModuleList([
            ResEncoderBlock(384, 384, n_blocks, None) 
            for _ in range(latent_layers)
        ])

    def forward(self, x):
        for layer in self.latent_blocks:
            x = layer(x)
        return x

class DJCMM(nn.Module):
    def __init__(self, in_channels, n_blocks, latent_layers, svs=False, window_length=1024, n_class=360):
        super(DJCMM, self).__init__()
        self.bn = nn.BatchNorm2d(window_length // 2 + 1, momentum=0.01)
        self.pe_encoder = Encoder(in_channels, n_blocks)
        self.pe_latent = LatentBlocks(n_blocks, latent_layers)
        self.pe_decoder = PE_Decoder(n_blocks, window_length=window_length, n_class=n_class)
        self.svs = svs

        if svs:
            self.svs_encoder = Encoder(in_channels, n_blocks)
            self.svs_latent = LatentBlocks(n_blocks, latent_layers)
            self.svs_decoder = SVS_Decoder(in_channels, n_blocks)

        init_bn(self.bn)

    def spec(self, x, spec_m):
        bs, c, time_steps, freqs_steps = x.shape
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)
        mask_spec = x[:, :, 0, :, :].sigmoid()
        linear_spec = x[:, :, 3, :, :]
        out_spec = (spec_m.detach() * mask_spec + linear_spec).relu()
        return out_spec

    def forward(self, spec):
        x = self.bn(spec.transpose(1, 3)).transpose(1, 3)[..., :-1]

        if self.svs:
            x, concat_tensors = self.svs_encoder(x)
            x = self.svs_decoder(self.svs_latent(x), concat_tensors)
            x = self.spec(nn.functional.pad(x, pad=(0, 1)), spec)[..., :-1]

        x, concat_tensors = self.pe_encoder(x)
        pe_out = self.pe_decoder(self.pe_latent(x), concat_tensors)
        return pe_out

# ==================== DJCM Predictor ====================
class DJCM:
    def __init__(
        self, 
        model_path, 
        device="cpu", 
        is_half=False, 
        onnx=False, 
        svs=False, 
        providers=["CPUExecutionProvider"], 
        batch_size=1, 
        segment_len=5.12, 
        kernel_size=3
    ):
        super(DJCM, self).__init__()
        if svs: 
            WINDOW_LENGTH = 2048
        self.onnx = onnx

        if self.onnx:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            model = DJCMM(1, 1, 1, svs=svs, window_length=WINDOW_LENGTH, n_class=N_CLASS)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.eval()
            if is_half: 
                model = model.half()
            self.model = model.to(device)

        self.batch_size = batch_size
        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.seg_frames = int(self.seg_len // int(SAMPLE_RATE // 100))

        self.device = device
        self.is_half = is_half
        self.kernel_size = kernel_size

        self.spec_extractor = Spectrogram(int(SAMPLE_RATE // 100), WINDOW_LENGTH).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def spec2hidden(self, spec):
        if self.onnx:
            spec = spec.cpu().numpy().astype(np.float32)
            hidden = torch.as_tensor(
                self.model.run(
                    [self.model.get_outputs()[0].name], 
                    {self.model.get_inputs()[0].name: spec}
                )[0], 
                device=self.device
            )
        else:
            if self.is_half: 
                spec = spec.half()
            hidden = self.model(spec)
        return hidden

    def infer_from_audio(self, audio, thred=0.03):
        if torch.is_tensor(audio): 
            audio = audio.cpu().numpy()
        if audio.ndim > 1: 
            audio = audio.squeeze()

        with torch.no_grad():
            padded_audio = self.pad_audio(audio)
            hidden = self.inference(padded_audio)[:(audio.shape[-1] // int(SAMPLE_RATE // 100) + 1)]
            f0 = self.decode(hidden.squeeze(0).cpu().numpy(), thred)
            if self.kernel_size is not None: 
                f0 = medfilt(f0, kernel_size=self.kernel_size)
            return f0
        
    def infer_from_audio_with_pitch(self, audio, thred=0.03, f0_min=50, f0_max=1100):
        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience, todo_cents_mapping = [], []
        starts = center - 4
        ends = center + 5

        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])

        todo_salience = np.array(todo_salience)
        devided = np.sum(todo_salience * np.array(todo_cents_mapping), 1) / np.sum(todo_salience, 1)
        devided[np.max(salience, axis=1) <= thred] = 0
        return devided
        
    def decode(self, hidden, thred=0.03):
        f0 = 10 * (2 ** (self.to_local_average_cents(hidden, thred=thred) / 1200))
        f0[f0 == 10] = 0
        return f0

    def pad_audio(self, audio):
        audio_len = audio.shape[-1]
        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = int(seg_nums * self.seg_len - audio_len + self.seg_len // 2)

        left_pad = np.zeros(int(self.seg_len // 4), dtype=np.float32)
        right_pad = np.zeros(int(pad_len - self.seg_len // 4), dtype=np.float32)
        padded_audio = np.concatenate([left_pad, audio, right_pad], axis=-1)

        segments = [
            padded_audio[start: start + int(self.seg_len)] 
            for start in range(0, len(padded_audio) - int(self.seg_len) + 1, int(self.seg_len // 2))
        ]

        segments = np.stack(segments, axis=0)
        segments = torch.from_numpy(segments).unsqueeze(1).to(self.device)
        return segments

    def inference(self, segments):
        hidden_segments = torch.cat([
            self.spec2hidden(self.spec_extractor(segments[i:i + self.batch_size].float()))
            for i in range(0, len(segments), self.batch_size)
        ], dim=0)

        hidden = torch.cat([
            seg[self.seg_frames // 4: int(self.seg_frames * 0.75)]
            for seg in hidden_segments
        ], dim=0)

        return hidden
