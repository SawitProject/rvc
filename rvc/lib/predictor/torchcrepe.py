import torch
import librosa
import functools
import scipy.stats

import numpy as np

CENTS_PER_BIN, MAX_FMAX, PITCH_BINS, SAMPLE_RATE, WINDOW_SIZE = 20, 2006, 360, 16000, 1024  

def mean(signals, win_length=9):
    assert signals.dim() == 2

    signals = signals.unsqueeze(1)
    mask = ~torch.isnan(signals)
    padding = win_length // 2

    ones_kernel = torch.ones(signals.size(1), 1, win_length, device=signals.device)
    avg_pooled = torch.nn.functional.conv1d(torch.where(mask, signals, torch.zeros_like(signals)), ones_kernel, stride=1, padding=padding) / torch.nn.functional.conv1d(mask.float(), ones_kernel, stride=1, padding=padding).clamp(min=1) 
    avg_pooled[avg_pooled == 0] = float("nan")

    return avg_pooled.squeeze(1)

def median(signals, win_length):
    assert signals.dim() == 2

    signals = signals.unsqueeze(1)
    mask = ~torch.isnan(signals)
    padding = win_length // 2

    x = torch.nn.functional.pad(torch.where(mask, signals, torch.zeros_like(signals)), (padding, padding), mode="reflect")
    mask = torch.nn.functional.pad(mask.float(), (padding, padding), mode="constant", value=0)

    x = x.unfold(2, win_length, 1)
    mask = mask.unfold(2, win_length, 1)

    x = x.contiguous().view(x.size()[:3] + (-1,))
    mask = mask.contiguous().view(mask.size()[:3] + (-1,))

    x_sorted, _ = torch.sort(torch.where(mask.bool(), x.float(), float("inf")).to(x), dim=-1)

    median_pooled = x_sorted.gather(-1, ((mask.sum(dim=-1) - 1) // 2).clamp(min=0).unsqueeze(-1).long()).squeeze(-1)
    median_pooled[torch.isinf(median_pooled)] = float("nan")

    return median_pooled.squeeze(1)

class CREPE_MODEL(torch.nn.Module):
    def __init__(self, model='full'):
        super().__init__()
        in_channels = {"full": [1, 1024, 128, 128, 128, 256], "large": [1, 768, 96, 96, 96, 192], "medium": [1, 512, 64, 64, 64, 128], "small": [1, 256, 32, 32, 32, 64], "tiny": [1, 128, 16, 16, 16, 32]}[model]
        out_channels = {"full": [1024, 128, 128, 128, 256, 512], "large": [768, 96, 96, 96, 192, 384], "medium": [512, 64, 64, 64, 128, 256], "small": [256, 32, 32, 32, 64, 128], "tiny": [128, 16, 16, 16, 32, 64]}[model]
        self.in_features = {"full": 2048, "large": 1536, "medium": 1024, "small": 512, "tiny": 256}[model]

        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]
        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d, eps=0.0010000000474974513, momentum=0.0)

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_sizes[0], stride=strides[0])
        self.conv1_BN = batch_norm_fn(num_features=out_channels[0])

        self.conv2 = torch.nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv2_BN = batch_norm_fn(num_features=out_channels[1])

        self.conv3 = torch.nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_sizes[2], stride=strides[2])
        self.conv3_BN = batch_norm_fn(num_features=out_channels[2])

        self.conv4 = torch.nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_sizes[3], stride=strides[3])
        self.conv4_BN = batch_norm_fn(num_features=out_channels[3])

        self.conv5 = torch.nn.Conv2d(in_channels=in_channels[4], out_channels=out_channels[4], kernel_size=kernel_sizes[4], stride=strides[4])
        self.conv5_BN = batch_norm_fn(num_features=out_channels[4])

        self.conv6 = torch.nn.Conv2d(in_channels=in_channels[5], out_channels=out_channels[5], kernel_size=kernel_sizes[5], stride=strides[5])
        self.conv6_BN = batch_norm_fn(num_features=out_channels[5])
        
        self.classifier = torch.nn.Linear(in_features=self.in_features, out_features=PITCH_BINS)

    def forward(self, x, embed=False):
        x = self.embed(x)
        if embed: return x
        return torch.sigmoid(self.classifier(self.layer(x, self.conv6, self.conv6_BN).permute(0, 2, 1, 3).reshape(-1, self.in_features)))

    def embed(self, x):
        x = x[:, None, :, None]
        return self.layer(self.layer(self.layer(self.layer(self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254)), self.conv2, self.conv2_BN), self.conv3, self.conv3_BN), self.conv4, self.conv4_BN), self.conv5, self.conv5_BN)

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        return torch.nn.functional.max_pool2d(batch_norm(torch.nn.functional.relu(conv(torch.nn.functional.pad(x, padding)))), (2, 1), (2, 1))

class CREPE:
    def __init__(self, model_path, model_size="full", hop_length=512, batch_size=None, f0_min=50, f0_max=1100, device=None, sample_rate=16000, return_periodicity=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hop_length = hop_length
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.return_periodicity = return_periodicity
        model = CREPE_MODEL(model_size)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()
        self.model = model.to(device)

    def bins_to_frequency(self, bins):
        if str(bins.device).startswith("ocl"): bins = bins.to(torch.float32)

        cents = CENTS_PER_BIN * bins + 1997.3794084376191
        return 10 * 2 ** ((cents + cents.new_tensor(scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size()))) / 1200)

    def frequency_to_bins(self, frequency, quantize_fn=torch.floor):
        return quantize_fn(((1200 * torch.log2(frequency / 10)) - 1997.3794084376191) / CENTS_PER_BIN).int()

    def viterbi(self, logits):
        if not hasattr(self, 'transition'):
            xx, yy = np.meshgrid(range(360), range(360))
            transition = np.maximum(12 - abs(xx - yy), 0)
            self.transition = transition / transition.sum(axis=1, keepdims=True)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=1)

        bins = torch.tensor(np.array([librosa.sequence.viterbi(sequence, self.transition).astype(np.int64) for sequence in probs.cpu().numpy()]), device=probs.device)
        return bins, self.bins_to_frequency(bins)
    
    def preprocess(self, audio, pad=True):
        hop_length = (self.sample_rate // 100) if self.hop_length is None else self.hop_length

        if self.sample_rate != SAMPLE_RATE:
            audio = torch.tensor(librosa.resample(audio.detach().cpu().numpy().squeeze(0), orig_sr=self.sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_vhq"), device=audio.device).unsqueeze(0)
            hop_length = int(hop_length * SAMPLE_RATE / self.sample_rate)

        if pad:
            total_frames = 1 + int(audio.size(1) // hop_length)
            audio = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        else: total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

        batch_size = total_frames if self.batch_size is None else self.batch_size

        for i in range(0, total_frames, batch_size):
            frames = torch.nn.functional.unfold(audio[:, None, None, max(0, i * hop_length):min(audio.size(1), (i + batch_size - 1) * hop_length + WINDOW_SIZE)], kernel_size=(1, WINDOW_SIZE), stride=(1, hop_length))
            
            if self.device.startswith("ocl"):
                frames = frames.transpose(1, 2).contiguous().reshape(-1, WINDOW_SIZE).to(self.device)
            else:
                frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE).to(self.device)

            frames -= frames.mean(dim=1, keepdim=True)
            frames /= torch.max(torch.tensor(1e-10, device=frames.device), frames.std(dim=1, keepdim=True))

            yield frames

    def periodicity(self, probabilities, bins):
        probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
        periodicity = probs_stacked.gather(1, bins.reshape(-1, 1).to(torch.int64))
        
        return periodicity.reshape(probabilities.size(0), probabilities.size(2))

    def postprocess(self, probabilities):
        probabilities = probabilities.detach()
        probabilities[:, :self.frequency_to_bins(torch.tensor(self.f0_min))] = -float('inf')
        probabilities[:, self.frequency_to_bins(torch.tensor(self.f0_max), torch.ceil):] = -float('inf')

        bins, pitch = self.viterbi(probabilities)

        if not self.return_periodicity: return pitch
        return pitch, self.periodicity(probabilities, bins)

    def compute_f0(self, audio, pad=True):
        results = []

        for frames in self.preprocess(audio, pad):
            with torch.no_grad():
                model = self.model(
                    frames, 
                    embed=False
                ).reshape(audio.size(0), -1, PITCH_BINS).transpose(1, 2)

            result = self.postprocess(model)
            results.append((result[0].to(audio.device), result[1].to(audio.device)) if isinstance(result, tuple) else result.to(audio.device))
        
        if self.return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)
        
        return torch.cat(results, 1)