# RVC (Retrieval-based Voice Conversion) Documentation

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Requirements](#requirements)
- [Python and NumPy Compatibility](#python-and-numpy-compatibility)
- [Features](#features)
- [Command-Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Google Colab](#google-colab)
- [Architecture](#architecture)
- [Bug Fix Changelog](#bug-fix-changelog)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

RVC (Retrieval-based Voice Conversion) is a simple, high-quality voice conversion tool focused on simplicity and ease of use. It enables high-quality voice conversion using machine learning models with various pitch extraction methods and advanced features.

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- FFmpeg installed on your system

### Installation Methods

#### Option 1: Install from Git (Recommended)
```bash
pip install git+https://github.com/SawitProject/rvc.git
```

#### Option 2: Install from Source (Development)
```bash
git clone https://github.com/SawitProject/rvc.git
cd rvc
pip install -e .
```

## Requirements

### System Requirements
- Operating System: Windows, macOS, or Linux
- Memory: At least 4GB RAM (8GB+ recommended for GPU processing)
- Storage: At least 500MB free space for installation
- Audio: Microphone for input, speakers/headphones for output

### Software Requirements
- Python 3.10-3.12
- FFmpeg (for audio processing)
- Compatible GPU (optional, for accelerated processing)

### Python Dependencies
- torch >= 2.3.1
- torchvision >= 0.18.1
- torchaudio >= 2.3.1
- numpy >= 1.25.2 (compatible with 2.0+)
- librosa >= 0.10.2
- soundfile >= 0.13.0
- faiss-cpu >= 1.7.3
- einops >= 0.8.0
- praat-parselmouth
- scipy >= 1.15.0
- omegaconf >= 2.0.6
- transformers >= 4.49.0
- And more (see pyproject.toml for complete list)

## Python and NumPy Compatibility

### Supported Python Versions
This project officially supports Python versions 3.10, 3.11, and 3.12. The code has been tested and verified to work correctly across all these versions.

### NumPy Compatibility
The project has been updated to be compatible with newer NumPy versions (2.0+), resolving compatibility issues that previously existed with Python 3.11+ installations. Key improvements include:

- Replaced deprecated `np.rint()` with `np.round()` for better compatibility
- Added fallback import mechanisms for NumPy functions that may vary between versions
- Maintained backward compatibility with older NumPy versions where possible

### Known Issues
- NumPy 2.0 introduced some API changes that required code adjustments
- Some older NumPy functions have been deprecated and replaced with newer equivalents

## Features

### Pitch Extraction Methods
RVC supports over 20 different pitch extraction methods including PM, DIO, CREPE (with multiple model sizes), FCPE, RMVPE, Harvest, YIN, PYIN, SWIPE, and DJCM. Additionally, hybrid methods allow combining multiple extractors (e.g., `hybrid[rmvpe+fcpe]`) for more robust pitch detection.

### Advanced Features
- **Formant Shifting**: Modify vocal tract characteristics independently of pitch using STFT-based pitch shifting
- **Noise Reduction**: Clean up audio output using spectral gating noise reduction
- **Autotune**: Automatic pitch correction with adjustable strength (0.0-1.0, default 1.0)
- **Batch Processing**: Process multiple audio files at once via directory input
- **Model Indexing**: Use .index files with FAISS for better voice quality preservation through feature retrieval
- **Proposal Pitch**: Automatically calculate optimal pitch shift based on source audio characteristics
- **Hybrid F0**: Combine multiple pitch extraction methods for improved robustness

### Embedder Models
- **ContentVec** (default): General-purpose voice embedding model
- **HuBERT Base**: Multilingual support including Japanese, Korean, Chinese, and Portuguese variants
- **SPIN**: Speaker Independent Network

### Vocoder Options
- **Default**: HiFi-GAN NSF (Neural Source Filter) for f0 models
- **RefineGAN**: Alternative high-quality vocoder
- **MRF-HiFi-GAN**: Multi-Receptive Field HiFi-GAN
- **HiFi-GAN**: Standard vocoder for non-f0 models

## Command-Line Interface

### Basic Usage
```bash
rvc -i input.wav -o output.wav -m model.pth
```

### Options

#### Basic Options
- `-i, --input`: Path to input audio file or directory (required)
- `-o, --output`: Path to output audio file (default: ./output.wav)
- `-m, --model`: Path to .pth model file (required)

#### Common Options
- `-p, --pitch`: Pitch shift in semitones (default: 0)
- `-f0, --f0_method`: F0 prediction method (default: rmvpe)
- `-idx, --index`: Path to .index file
- `-ir, --index_rate`: Index rate for feature retrieval (default: 0.5)
- `-em, --embedder`: Embedder model name (default: contentvec_base)

#### Advanced Options
- `-split, --split_audio`: Split audio into chunks for processing
- `-clean, --clean_audio`: Apply noise reduction to output
- `-cs, --clean_strength`: Noise reduction strength 0.0-1.0 (default: 0.7)
- `-fa, --f0_autotune`: Enable F0 autotune
- `-fas, --f0_autotune_strength`: Autotune strength 0.0-1.0 (default: 1.0)
- `-fs, --formant_shifting`: Enable formant shifting
- `-fq, --formant_qfrency`: Formant frequency parameter (default: 0.8)
- `-ft, --formant_timbre`: Formant timbre parameter (default: 0.8)
- `-pp, --proposal_pitch`: Enable proposal pitch
- `-ppt, --proposal_pitch_threshold`: Proposal pitch threshold (default: 255.0)
- `-fr, --filter_radius`: Filter radius for pitch extraction (default: 3)
- `-hl, --hop_length`: Hop length for pitch extraction (default: 64)
- `-rs, --resample_sr`: Resample output audio sample rate (default: 0, no resampling)
- `-fmt, --format`: Output format: wav, flac, mp3, ogg (default: wav)

### Example Usage
```bash
# Simple conversion
rvc -i input.wav -o output.wav -m model.pth -p 12

# Batch conversion from directory
rvc -i ./audio_folder -m model.pth -p 12 -f0 rmvpe

# With index file and autotune
rvc -i input.wav -m model.pth -idx model.index -ir 0.75 -fa

# With formant shifting
rvc -i input.wav -o output.wav -m model.pth -fs -fq 0.9 -ft 0.7

# Advanced processing with multiple options
rvc -i input.wav -o output.wav -m model.pth -p -5 -f0 rmvpe -idx model.index -ir 0.8 -fa -fas 0.8 -clean

# Using CREPE large model for highest quality
rvc -i input.wav -o output.wav -m model.pth -f0 crepe-large

# Hybrid method combining RMVPE and FCPE
rvc -i input.wav -o output.wav -m model.pth -f0 "hybrid[rmvpe+fcpe]"
```

## Python API

### Basic Inference
```python
from rvc.infer.infer import run_inference_script
from rvc.lib.config import Config

config = Config()

run_inference_script(
    config=config,
    input_path="input.wav",
    output_path="output.wav",
    pth_path="model.pth",
    pitch=12,
    f0_method="rmvpe",
)
```

### Using VoiceConverter Directly
```python
from rvc.infer.cli import VoiceConverter
from rvc.lib.config import Config

config = Config()
converter = VoiceConverter(config, model_path="model.pth", sid=0)

converter.convert_audio(
    audio_input_path="input.wav",
    audio_output_path="output.wav",
    index_path="model.index",
    embedder_model="contentvec_base",
    pitch=12,
    f0_method="rmvpe",
    index_rate=0.5,
    volume_envelope=1.0,
    protect=0.5,
    hop_length=64,
    filter_radius=3,
    export_format="wav",
)
```

### Pitch Extraction
```python
from rvc.lib.predictor.generator import Generator

gen = Generator(sample_rate=16000, hop_length=160, f0_min=50, f0_max=1100, is_half=False, device="cpu")
f0_mel, f0_hz = gen.calculator(
    f0_method="rmvpe",
    x=audio_data,
    f0_up_key=0,
    p_len=None,
    filter_radius=3,
    f0_autotune=True,
    f0_autotune_strength=0.8,
)
```

### Autotune
```python
from rvc.utils import Autotune

ref_freqs = [49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41,
             87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83]
autotuner = Autotune(ref_freqs)
adjusted_f0 = autotuner.autotune_f0(original_f0, f0_autotune_strength=0.8)
```

## Google Colab

A ready-to-run Colab notebook is available for quick testing without local installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SawitProject/rvc/blob/main/colab/rvc_demo.ipynb)

The notebook includes:
- Automatic installation of RVC and dependencies
- Model download from HuggingFace
- Simple voice conversion demo
- Batch processing example

## Architecture

### Project Structure
```
rvc/
  __init__.py
  utils.py              # Utility functions (audio loading, GPU cache, downloads)
  var.py                # Global variables (f0 method list)
  lib/
    algorithm/          # Core ML algorithm components
      commons.py        # Shared utilities, weight init
      attentions.py     # Attention mechanisms
      encoders.py       # Text and posterior encoders
      residuals.py      # Residual coupling blocks
      synthesizers.py   # Main Synthesizer model
      normalization.py  # Normalization layers
      modules.py        # Building block modules
    backend/            # Hardware backend support
      opencl.py         # AMD OpenCL support
      rms.py            # RMS energy extraction
      stftpitchshift.py # STFT pitch shifting
      zulda.py          # Additional backend
    embedders/          # Voice embedding models
      fairseq.py        # Fairseq/HuBERT embedder
    generators/         # Vocoder generators
      hifigan.py        # HiFi-GAN vocoder
      nsf_hifigan.py    # NSF HiFi-GAN vocoder
      refinegan.py      # RefineGAN vocoder
      mrf_hifigan.py    # MRF HiFi-GAN vocoder
    onnx/               # ONNX export support
      onnx_export.py    # Model export to ONNX
      wrapper.py        # ONNX inference wrapper
    predictor/          # Pitch estimation algorithms
      generator.py      # F0 computation dispatcher
      rmvpe.py          # RMVPE pitch estimator
      torchfcpe.py      # FCPE pitch estimator
      torchcrepe.py     # CREPE pitch estimator
      djcm.py           # DJCM pitch estimator
      pyworld.py        # PyWorld (DIO/Harvest) estimator
      swipe.py          # SWIPE pitch estimator
    config.py           # Configuration (device, paths)
  infer/                # Inference pipeline
    cli.py              # Command-line interface
    infer.py            # Python API entry point
    pipeline.py         # Voice conversion pipeline
  tools/                # Utility tools
    cut.py              # Audio splitting/restoration
    gdown.py            # Google Drive downloader
    noisereduce.py      # Noise reduction
  tts/                  # Text-to-speech integration
    infer.py            # TTS inference
```

### Inference Pipeline Flow
1. **Audio Loading**: Load and resample input audio to 16kHz
2. **Embedding Extraction**: Extract voice features using HuBERT/ContentVec
3. **Feature Retrieval** (optional): Enhance features using FAISS index
4. **Pitch Estimation**: Extract F0 contour using selected method
5. **Autotune** (optional): Snap pitch to nearest musical notes
6. **Voice Conversion**: Run synthesizer model with features + pitch
7. **Post-processing**: Volume envelope matching, noise reduction, resampling

## Bug Fix Changelog

### Critical Fixes

#### BUG 1: HiFi-GAN Generator `forward()` inaccessible
- **File**: `rvc/lib/generators/hifigan.py`
- **Description**: The `forward()` method was defined as a nested function inside `__init__()` instead of as a class method. PyTorch calls `model.forward()` when invoking `model(x)`, which would fail with `NotImplementedError` since no class-level `forward` existed. This made the entire HiFi-GAN vocoder non-functional.
- **Fix**: Dedented the `forward()` method so it is a proper class method at the correct indentation level.

#### BUG 2: `SynthesizerONNX` class doesn't exist
- **File**: `rvc/lib/onnx/onnx_export.py`
- **Description**: The ONNX export module imported `SynthesizerONNX` from `synthesizers.py`, but only `Synthesizer` was defined there. This caused an `ImportError` at module load time, breaking all ONNX export functionality.
- **Fix**: Changed the import to `from rvc.lib.algorithm.synthesizers import Synthesizer as SynthesizerONNX`, since the same `Synthesizer` class can serve both purposes.

#### BUG 3: `SelfAttention.__init__` missing `use_norm` parameter
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: `CFNEncoderLayer` called `SelfAttention(dim=dim_model, heads=num_heads, causal=False, use_norm=use_norm, ...)` but `SelfAttention.__init__` did not accept `use_norm`. This raised `TypeError` whenever the FCPE model was instantiated.
- **Fix**: Added `use_norm=None` parameter to `SelfAttention.__init__` signature.

#### BUG 4: `FastAttention.causal_linear_fn` doesn't exist
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: When `self.causal=True`, `forward()` referenced `self.causal_linear_fn`, which was never defined. This would raise `AttributeError` if causal attention was ever used.
- **Fix**: Implemented `_causal_linear_fn` as a static method using cumulative sum-based linear attention.

#### BUG 5: `compute_f0()` method name splitting breaks CREPE and legacy methods
- **File**: `rvc/lib/predictor/generator.py`
- **Description**: `base_method = f0_method.split('-')[0]` was used for dict lookup, causing: `"mangio-crepe-tiny"` to resolve to `"mangio"` (not found, falls back to PM), `"crepe-tiny"` to resolve to `"crepe"` (not found, falls back to PM), and `"rmvpe-legacy"` to resolve to `"rmvpe"` (calling non-legacy function). All CREPE variants and legacy methods were silently broken.
- **Fix**: Removed the `base_method` splitting logic and used the full `f0_method` string for dict lookup.

#### BUG 6: Wrong import path for `noisereduce`
- **File**: `rvc/infer/cli.py`, `rvc/infer/infer.py`
- **Description**: Both files imported `from rvc.lib.tools.noisereduce import reduce_noise`, but `noisereduce.py` lives at `rvc/tools/noisereduce.py`, not `rvc/lib/tools/noisereduce.py`. This caused `ModuleNotFoundError` whenever `clean_audio` was enabled.
- **Fix**: Changed import to `from rvc.tools.noisereduce import reduce_noise`.

#### BUG 7: Wrong import in DJCM for OpenCL devices
- **File**: `rvc/lib/predictor/djcm.py`
- **Description**: `Spectrogram.forward()` imported `from main.library.backends.utils import STFT` for OpenCL devices, but this path does not exist in the project. The correct import should be from the project's own `opencl` backend module.
- **Fix**: Changed to `from rvc.lib.backend.opencl import STFT`.

#### BUG 8: `PYWORLD.__init__` opens a directory instead of a file
- **File**: `rvc/lib/predictor/pyworld.py`
- **Description**: `open(os.path.join("assets"), "rb")` attempted to open the `"assets"` directory as a file for `pickle.load()`, raising `IsADirectoryError` on Linux/macOS or `PermissionError` on Windows when the WORLD DLL was not yet extracted.
- **Fix**: Changed to `open(os.path.join("assets", "models", "world.bin"), "rb")` to correctly reference the pickle file containing the DLL data.

#### BUG 9: `VoiceConverter.cleanup()` double-deletes `self.net_g`
- **File**: `rvc/infer/cli.py`, `rvc/infer/infer.py`
- **Description**: The `cleanup()` method first deleted `self.net_g` inside the `if self.hubert_model is not None` block, then attempted `del self.net_g` again in a separate `if hasattr(self, 'net_g')` block. Since `net_g` was already deleted (along with other attributes) in the first block, `hasattr(self, 'net_g')` could still be True in edge cases, leading to `AttributeError`.
- **Fix**: Removed the redundant `if hasattr(self, 'net_g'): del self.net_g` block, keeping only the proper cleanup within the `hubert_model` check.

#### BUG 10: DJCM `WINDOW_LENGTH` local variable causes `UnboundLocalError`
- **File**: `rvc/lib/predictor/djcm.py`
- **Description**: Line `WINDOW_LENGTH = 2048` inside an `if svs:` block shadowed the module-level constant. Python treats any variable assigned anywhere in a function as local throughout that function, so when `svs=False`, accessing `WINDOW_LENGTH` raised `UnboundLocalError`.
- **Fix**: Used a local variable `window_length = 2048 if svs else WINDOW_LENGTH` instead of reassigning the module constant.

### Significant Fixes

#### BUG 11: `f0_autotune_strength` defaults to `False`
- **File**: `rvc/infer/pipeline.py`
- **Description**: `f0_autotune_strength=False` meant `False` evaluates to `0` in numeric context, so `autotune_f0()` returned the original f0 unmodified (since `if f0_autotune_strength <= 0: return f0`). Autotune had no effect with the default parameter.
- **Fix**: Changed default to `f0_autotune_strength=1.0`.

#### BUG 12: `gaussian_blurred_cent` operator precedence bug
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: The expression `torch.exp(...) * (cents > 0.1) & (cents < ...)` evaluated as `(torch.exp(...) * (cents > 0.1)) & (cents < ...)` because `*` has higher precedence than `&`. The intended behavior was `torch.exp(...) * ((cents > 0.1) & (cents < ...))`, producing a float probability tensor for `F.binary_cross_entropy`.
- **Fix**: Added explicit parentheses: `* ((cents > 0.1) & (cents < ...)).float()`.

#### BUG 13: Both "argmax" and "local_argmax" decoders map to the same function
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: In `CFNaiveMelPE.infer()`, when `decoder == "argmax"`, the code assigned `cents = self.latent2cents_local_decoder` instead of `self.latent2cents_decoder`, making the argmax decoder option non-functional.
- **Fix**: Changed to `if decoder == "argmax": cents = self.latent2cents_decoder`.

#### BUG 14: `gelu_accurate` returns `None` on subsequent calls
- **File**: `rvc/lib/embedders/fairseq.py`
- **Description**: The `return` statement was inside the `if not hasattr(gelu_accurate, "_a")` block. On the first call, it set `_a` and returned correctly. On subsequent calls, it skipped the `if` block and had no `return` statement, returning `None`.
- **Fix**: Moved the `return` statement outside the `if` block so it executes on every call.

#### BUG 15: `assert src_len, key_bsz == value.shape[:2]` always True
- **File**: `rvc/lib/embedders/fairseq.py`
- **Description**: Python parsed `assert src_len, key_bsz == value.shape[:2]` as a tuple assertion: `assert (src_len,)` with the comparison as the error message. Since `src_len` (a dimension) is always truthy, the assertion was always True regardless of whether the shapes matched.
- **Fix**: Changed to `assert src_len == value.shape[0] and key_bsz == value.shape[1]`.

#### BUG 16: `SelfAttention.forward` doesn't handle `cross_attend=True`
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: When `cross_attend` was True (i.e., context was provided), the code just `pass`ed, leaving `out` undefined. Then `attn_outs.append(out)` raised `UnboundLocalError`.
- **Fix**: Changed the `pass` to `out = self.fast_attention(q, k, v)` to properly compute cross-attention output.

### Minor Fixes

#### BUG 17: Duplicate `nn` import in DJCM
- **File**: `rvc/lib/predictor/djcm.py`
- **Fix**: Consolidated `import torch.nn as nn` and `from torch import nn` into a single `from torch import nn`.

#### BUG 18: Duplicate `self.resample_kernel = {}` in RMVPE
- **File**: `rvc/lib/predictor/rmvpe.py`
- **Fix**: Removed the duplicate assignment.

#### BUG 19: `MelModule.__call__` overrides `nn.Module.__call__`
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: Both `MelModule` and `Wav2MelModule` overrode `__call__` directly instead of implementing `forward`. This bypassed PyTorch's `nn.Module.__call__` mechanism (hooks, `torch.no_grad()` context, etc.).
- **Fix**: Renamed `__call__` to `forward` in both classes.

#### BUG 20: `pytorch_ocl != None` instead of `is not None`
- **File**: `rvc/lib/backend/opencl.py`
- **Fix**: Changed to `pytorch_ocl is not None` per PEP 8.

#### BUG 21: Redundant import fallback in `stftpitchshift.py`
- **File**: `rvc/lib/backend/stftpitchshift.py`
- **Description**: Both `try` and `except` blocks imported `sliding_window_view` from the same path, making the fallback useless.
- **Fix**: Changed the fallback to set `sliding_window_view = None`.

#### BUG 22: `os.path.join("assets")` is misleading
- **File**: `rvc/lib/predictor/pyworld.py`
- **Fix**: Simplified to `"assets"` since `os.path.join` with a single argument just returns that argument.

#### BUG 23: Cookie loading assumes list-of-pairs format
- **File**: `rvc/tools/gdown.py`
- **Description**: `for k, v in cookies:` assumed the JSON file stores cookies as a list of pairs `[[key, val], ...]`. Standard browser cookie exports typically use dict format `{"key": val}`.
- **Fix**: Added `isinstance(cookies, dict)` check to handle both dict and list formats.

## Troubleshooting

### Common Issues

#### Installation Issues
- **Problem**: Installation fails due to dependency conflicts
- **Solution**: Create a fresh virtual environment and install dependencies separately:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate     # Windows
  pip install git+https://github.com/SawitProject/rvc.git
  ```

- **Problem**: PyTorch installation fails
- **Solution**: Install PyTorch separately using the official PyTorch installation guide before installing RVC:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

#### Runtime Issues
- **Problem**: Out of memory errors during inference
- **Solution**: Reduce audio length, use CPU instead of GPU, or use the `-split` flag to process in chunks

- **Problem**: Audio format not supported
- **Solution**: Convert audio to WAV format using FFmpeg:
  ```bash
  ffmpeg -i input.mp3 input.wav
  ```

- **Problem**: `ModuleNotFoundError: No module named 'rvc.lib.tools.noisereduce'`
- **Solution**: This bug has been fixed. Update to the latest version.

- **Problem**: CREPE or legacy f0 methods silently produce wrong results
- **Solution**: This bug has been fixed. Update to the latest version.

- **Problem**: Autotune has no effect
- **Solution**: This bug has been fixed. The default `f0_autotune_strength` is now `1.0`. Make sure you're using the latest version.

#### NumPy Compatibility Issues
- **Problem**: `AttributeError: module 'numpy' has no attribute 'rint'`
- **Solution**: Upgrade to the latest version of this package which uses `np.round()` instead

### Performance Tips
- Use GPU acceleration when available for faster processing
- Process audio in smaller chunks for large files using `-split`
- Choose appropriate pitch extraction methods based on quality/speed requirements:
  - Fastest: PM, DIO
  - Balanced: Harvest, SWIPE
  - Highest Quality: RMVPE, CREPE (larger models)
- Use `-hl 128` (default hop length) for best quality; increase for faster processing

## Contributing

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/rvc.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
5. Install in development mode: `pip install -e .`

### Code Standards
- Follow PEP 8 style guidelines
- Write docstrings for all public functions and classes
- Add type hints where appropriate
- Include tests for new functionality

### Pull Request Process
1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Add tests if applicable
4. Update documentation as needed
5. Submit a pull request to the main branch

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on the GitHub repository with:
- Detailed description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)

## Acknowledgments

- Original Vietnamese RVC project for the foundational algorithm
- Various open-source libraries that make this project possible
- The community for contributions and feedback
