import os
import gc
import sys
import torch
import codecs
import librosa
import requests
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from tqdm import tqdm  # Imported for progress bar

sys.path.append(os.getcwd())

from rvc.lib.backend import opencl



def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())

def clear_gpu_cache():
    gc.collect()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif opencl.is_available(): opencl.pytorch_ocl.empty_cache()

def HF_download_file(url, output_path=None):
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    output_path = os.path.basename(url) if output_path is None else (os.path.join(output_path, os.path.basename(url)) if os.path.isdir(output_path) else output_path)
    
    # Ensure the directory exists before attempting to write the file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    response = requests.get(url, stream=True, timeout=300)

    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        
        # Initialize progress bar
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path))
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        return output_path
    else: 
        raise ValueError(f"Failed to download. Status Code: {response.status_code}")

def check_predictors(method):
    def download(predictors):
        PREDICTOR_MODEL = os.path.join(os.getcwd(), "assets", "models", predictors)
        if not os.path.exists(os.path.join(PREDICTOR_MODEL): 
           # Ensure directory exists
           predictor_dir = os.path.dirname(PREDICTOR_MODEL)
           
           HF_download_file(os.path.join("https://huggingface.co/NeoPy/Ultimate-Models/resolve/main/predictors/", predictors), PREDICTOR_MODEL)

    model_dict = {
        **dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"), 
        **dict.fromkeys(["fcpe"], "fcpe.pt"), 
        **dict.fromkeys(["fcpe-legacy"], "fcpe_legacy.pt"), 
        **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"), 
        **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"), 
        **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"), 
        **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"), 
        **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"), 
    }

    if method in model_dict: download(model_dict[method])

def check_embedders(hubert):
    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base", "spin"]:
        hubert += ".pt"
        # FIXED: Removed the leading space in " models" -> "models"
        HUBERT_PATH = os.path.join(os.getcwd(), "assets", "models", hubert)
        if not os.path.exists(HUBERT_PATH): 
            HF_download_file(os.path.join("https://huggingface.co/NeoPy/Ultimate-Models/resolve/main/embedders/fairseq/", hubert), HUBERT_PATH)

def load_audio(file, sample_rate=16000, formant_shifting=False, formant_qfrency=0.8, formant_timbre=0.8):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(f"[ERROR] Not found audio: {file}")

        try:
            audio, sr = sf.read(file, dtype=np.float32)
        except:
            audio, sr = librosa.load(file, sr=None)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")

        if formant_shifting:
            from rvc.lib.backend.stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(audio, factors=1, quefrency=formant_qfrency * 1e-3, distortion=formant_timbre)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Error reading audio file: {e}")
    
    return audio.flatten()

class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            autotuned_f0[i] = freq + (min(self.note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

        return autotuned_f0
