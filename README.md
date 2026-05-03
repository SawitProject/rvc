<div align="center">

# Simple RVC

A simple, high-quality voice conversion tool focused on simplicity and ease of use.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SawitProject/rvc/blob/main/colab/rvc_demo.ipynb)
[![Python 3.10-3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


## Key Features

* **20+ pitch extraction methods**: pm, dio, crepe, fcpe, rmvpe, harvest, yin, pyin, swipe, djcm and more
* **Hybrid f0 methods**: combine multiple pitch extractors (e.g. `hybrid[rmvpe+fcpe]`) for robust results
* **Powerful CLI** for single-file and batch processing
* **Multiple embedder models**: contentvec, hubert (multilingual), spin
* **Advanced features**: formant shifting, noise reduction, autotune with adjustable strength, proposal pitch
* **REST API**: FastAPI-based HTTP server for integration into any application
* **ONNX export** for optimized inference
* **Multi-backend support**: NVIDIA CUDA, AMD OpenCL, Apple MPS, CPU fallback

## Quick Start

### Try it in Google Colab
Click the badge below to open a ready-to-run demo notebook — no local installation required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SawitProject/rvc/blob/main/colab/rvc_demo.ipynb)

### Install

```bash
pip install git+https://github.com/SawitProject/rvc.git
```

### CLI

```bash
rvc -i input.wav -o output.wav -m model.pth -p 12
```

### Python API

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

## Installation

**Requires Python 3.10-3.12 and FFmpeg installed**

### Option 1: Install from Git (Recommended)
```bash
pip install git+https://github.com/SawitProject/rvc.git
```

### Option 2: Install from Source (Development)
```bash
git clone https://github.com/SawitProject/rvc.git
cd rvc
pip install -e .
```

### Python Version Compatibility

This project supports Python 3.10, 3.11, and 3.12. The code has been updated to be compatible with newer numpy versions (2.0+) which resolves compatibility issues that previously existed with Python 3.11+ installations.

### Platform-Specific Notes

**NVIDIA GPU:**
- PyTorch with CUDA support is automatically installed
- Ensure you have CUDA drivers installed

**AMD GPU:**
- May require additional setup for OpenCL support
- Consider using ROCm PyTorch if available for your hardware

**CPU Only:**
- The package will automatically use CPU inference
- Note: CPU inference will be slower than GPU


## Command Line Interface

After installation, you can use the RVC CLI tool for voice conversion:

```bash
rvc -i input.wav -o output.wav -m model.pth
```

**Basic Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Path to input audio file or directory | (required) |
| `-o, --output` | Path to output audio file | `./output.wav` |
| `-m, --model` | Path to .pth model file | (required) |

**Common Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-p, --pitch` | Pitch shift in semitones | `0` |
| `-f0, --f0_method` | F0 prediction method | `rmvpe` |
| `-idx, --index` | Path to .index file | `None` |
| `-ir, --index_rate` | Index rate for feature retrieval | `0.5` |
| `-em, --embedder` | Embedder model name | `contentvec_base` |

**Advanced Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-split, --split_audio` | Split audio into chunks for processing | `False` |
| `-clean, --clean_audio` | Apply noise reduction to output | `False` |
| `-cs, --clean_strength` | Noise reduction strength (0.0-1.0) | `0.7` |
| `-fa, --f0_autotune` | Enable F0 autotune | `False` |
| `-fas, --f0_autotune_strength` | Autotune strength (0.0-1.0) | `1.0` |
| `-fs, --formant_shifting` | Enable formant shifting | `False` |
| `-fq, --formant_qfrency` | Formant quefrency | `0.8` |
| `-ft, --formant_timbre` | Formant timbre | `0.8` |
| `-pp, --proposal_pitch` | Enable proposal pitch | `False` |
| `-ppt, --proposal_pitch_threshold` | Proposal pitch threshold | `255.0` |
| `-fr, --filter_radius` | Filter radius for pitch extraction | `3` |
| `-hl, --hop_length` | Hop length for pitch extraction | `64` |
| `-rs, --resample_sr` | Resample output sample rate (0=disabled) | `0` |
| `-fmt, --format` | Output format (wav, flac, mp3, ogg) | `wav` |

### Available F0 Methods

| Method | Quality | Speed | Notes |
|--------|---------|-------|-------|
| `pm` | Low | Fastest | Parselmouth, good for quick testing |
| `dio` | Low | Fast | PyWorld DIO algorithm |
| `harvest` | Medium | Medium | PyWorld Harvest algorithm |
| `yin` | Medium | Medium | Librosa YIN |
| `pyin` | Medium | Medium | Librosa PYIN (probabilistic) |
| `swipe` | Medium | Medium | Sawtooth Waveform Inspired Pitch Estimator |
| `rmvpe` | High | Slow | Recommended for best quality |
| `rmvpe-legacy` | High | Slow | RMVPE with pitch filtering |
| `fcpe` | High | Slow | Full Cycle Pitch Estimation |
| `fcpe-legacy` | High | Slow | FCPE with legacy threshold |
| `crepe-tiny` | Medium | Medium | CREPE tiny model |
| `crepe-small` | Medium-High | Medium | CREPE small model |
| `crepe-medium` | High | Slow | CREPE medium model |
| `crepe-large` | High | Slower | CREPE large model |
| `crepe-full` | Highest | Slowest | CREPE full model |
| `mangio-crepe-*` | Varies | Varies | CREPE variants with different normalization |
| `djcm` | High | Slow | Deep Jungwoo Convolution Model |
| `hybrid[method1+method2]` | High | Slow | Combine methods (e.g. `hybrid[rmvpe+fcpe]`) |

### Example Usage

```bash
# Simple conversion
rvc -i input.wav -o output.wav -m model.pth -p 12

# Batch conversion from directory
rvc -i ./audio_folder -m model.pth -p 12 -f0 rmvpe

# With index file and autotune
rvc -i input.wav -m model.pth -idx model.index -ir 0.75 -fa

# With noise reduction and formant shifting
rvc -i input.wav -o output.wav -m model.pth -clean -fs -fq 0.9 -ft 0.7

# Using CREPE large for highest quality
rvc -i input.wav -o output.wav -m model.pth -f0 crepe-large

# Hybrid method combining RMVPE and FCPE
rvc -i input.wav -o output.wav -m model.pth -f0 "hybrid[rmvpe+fcpe]"
```

For more options, run:
```bash
rvc --help
```

## REST API

RVC includes a built-in REST API server powered by FastAPI, allowing you to integrate voice conversion into any application over HTTP.

### Starting the Server

```bash
# Using the CLI entry point
rvc-api --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn rvc.api.app:app --host 0.0.0.0 --port 8000
```

The interactive API docs are available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

### Quick Example

```bash
# 1. Load a voice model
curl -X POST http://localhost:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/model.pth"}'
# Response: {"model_id": "abc123def456", "model_path": "...", "version": "v2", ...}

# 2. Convert audio (upload file)
curl -X POST http://localhost:8000/api/v1/convert \
  -F "audio=@input.wav" \
  -F "model_id=abc123def456" \
  -F "pitch=12" \
  -F "f0_method=rmvpe" \
  -o output.wav

# 3. Convert audio (server file path)
curl -X POST http://localhost:8000/api/v1/convert/file \
  -H "Content-Type: application/json" \
  -d '{"model_id": "abc123def456", "input_path": "/path/to/input.wav", "pitch": 12}'

# 4. Unload model
curl -X DELETE http://localhost:8000/api/v1/models/abc123def456
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check & system status |
| `GET` | `/api/v1/config` | Current server configuration |
| `GET` | `/api/v1/methods` | List available F0 methods |
| `GET` | `/api/v1/embedders` | List available embedder models |
| `POST` | `/api/v1/models/load` | Load a voice model into memory |
| `GET` | `/api/v1/models` | List all loaded models |
| `GET` | `/api/v1/models/{model_id}` | Get info about a loaded model |
| `DELETE` | `/api/v1/models/{model_id}` | Unload a model from memory |
| `POST` | `/api/v1/convert` | Convert audio (file upload, returns audio stream) |
| `POST` | `/api/v1/convert/file` | Convert audio (server paths, returns output path) |

For full request/response schemas, see the interactive docs at `/docs`.

## Models

**Using CLI:**
- Specify the model path with the `-m` option
- Specify the index path with the `-idx` option (recommended for better voice quality)

**Model Download:**
Predictor and embedder models are automatically downloaded from HuggingFace when first used. No manual setup needed.

**Note:** Pre-trained RVC models (.pth files) can be downloaded from various sources. Ensure you have the right to use any model before converting audio with it.

## Recent Bug Fixes

The following bugs have been identified and fixed in this repository:

### Critical Fixes
- **HiFi-GAN Generator**: `forward()` method was incorrectly nested inside `__init__()`, making the vocoder completely non-functional
- **ONNX Export**: `SynthesizerONNX` class didn't exist, causing `ImportError` on import
- **FCPE Model**: `SelfAttention` was missing `use_norm` parameter, causing `TypeError` during model creation
- **F0 Method Lookup**: `compute_f0()` was splitting method names on `-`, causing all CREPE and legacy methods to silently fall back to PM
- **Noisereduce Import**: Wrong import path `rvc.lib.tools.noisereduce` (should be `rvc.tools.noisereduce`) — noise reduction was broken
- **DJCM OpenCL**: Wrong import `from main.library.backends.utils import STFT` — DJCM on OpenCL devices was broken
- **PyWorld DLL**: Was attempting to open the `assets` directory as a file instead of `assets/models/world.bin`
- **Memory Cleanup**: `VoiceConverter.cleanup()` double-deleted `net_g` causing `AttributeError`
- **DJCM Window Length**: `WINDOW_LENGTH` reassignment inside `if svs:` block caused `UnboundLocalError` when `svs=False`

### Significant Fixes
- **Autotune Strength**: `f0_autotune_strength` defaulted to `False` instead of `1.0`, making autotune ineffective by default
- **FCPE Gaussian Blur**: Operator precedence bug caused `*` to bind before `&`, producing boolean tensors instead of float probabilities
- **FCPE Decoder**: Both "argmax" and "local_argmax" decoder options mapped to the same decoder function
- **GELU Activation**: `gelu_accurate()` returned `None` on all calls after the first
- **Fairseq Assertion**: `assert src_len, key_bsz == value.shape[:2]` was parsed as a tuple assertion (always True)
- **Cross-Attention**: `SelfAttention.forward()` left `out` undefined when `cross_attend=True`

For the complete list, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Disclaimer
- The RVC project is developed for research, educational, and personal entertainment purposes. I do not encourage, nor do I take any responsibility for, any misuse of voice conversion technology for fraudulent purposes, identity impersonation, or violations of privacy or copyright belonging to any individual or organization.

- Users are solely responsible for how they use this software and must comply with the laws and regulations of the country in which they reside or operate.

- The use of voices of celebrities, real people, or public figures must be authorized or ensured not to violate any applicable laws, ethical standards, or the rights of the individuals involved.

- The author of this project holds no legal liability for any consequences arising from the use of this software.

## Terms of Use
- You must ensure that any audio content you upload and convert through this project does not infringe upon the intellectual property rights of any third party.

- This project must not be used for any illegal activity, including but not limited to fraud, harassment, or causing harm to others.

- You are fully responsible for any damages resulting from improper use of the product.

- I am not liable for any direct or indirect damages arising from the use of this project.

## Documentation

For detailed documentation, including API references, troubleshooting guides, and advanced usage, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Project-based construction

- **Algorithm: [Vietnamese RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)**
