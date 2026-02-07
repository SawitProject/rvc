# RVC (Retrieval-based Voice Conversion) Documentation

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Requirements](#requirements)
- [Python and NumPy Compatibility](#python-and-numpy-compatibility)
- [Features](#features)
- [Command-Line Interface](#command-line-interface)
- [API Reference](#api-reference)
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
RVC supports over 20 different pitch extraction methods including PM, DIO, CREPE (with multiple model sizes), FCPE, RMVPE, Harvest, YIN, PYIN, SWIPE, and DJCM.

### Advanced Features
- **Formant Shifting**: Modify vocal tract characteristics independently of pitch
- **Noise Reduction**: Clean up audio input and output
- **Autotune**: Automatic pitch correction with adjustable strength
- **Batch Processing**: Process multiple audio files at once
- **Model Indexing**: Use index files for better voice quality preservation

### Embedder Models
- ContentVec
- HuBERT (multiple language variants)
- SPIN (Speaker Independent Network)

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

#### Advanced Options
- `-split, --split_audio`: Split audio into chunks for processing
- `-clean, --clean_audio`: Apply noise reduction to output
- `-fa, --f0_autotune`: Enable F0 autotune
- `-fs, --formant_shifting`: Enable formant shifting
- `-fq, --formant_qfrency`: Formant frequency parameter (default: 0.8)
- `-ft, --formant_timbre`: Formant timbre parameter (default: 0.8)
- `-ats, --f0_autotune_strength`: Autotune strength (0.0-1.0, default: 1.0)
- `-fr, --filter_radius`: Filter radius for pitch extraction (default: 3)
- `-res, --resample_sr`: Resample output audio (default: 0, no resampling)
- `-rms, --rms_mix_rate`: RMS mix rate (default: 0.25)
- `-hop, --hop_length`: Hop length for pitch extraction (default: 128)

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
rvc -i input.wav -o output.wav -m model.pth -p -5 -f0 rmvpe -idx model.index -ir 0.8 -fa -ats 0.8 -clean
```

## API Reference

### Core Modules

#### rvc.utils
Utility functions for audio processing, model management, and helper functions.

#### rvc.infer
Main inference pipeline for voice conversion.

#### rvc.lib.predictor
Pitch extraction and prediction algorithms including RMVPE, FCPE, CREPE, and others.

#### rvc.lib.embedders
Embedding models for voice feature extraction.

#### rvc.lib.backend
Backend implementations for different processing methods.

### Key Classes and Functions

#### Generator Class
Handles pitch extraction and processing using various methods.

```python
from rvc.lib.predictor.generator import Generator

gen = Generator(sample_rate=16000, hop_length=160, f0_min=50, f0_max=1100)
f0_result = gen.calculator(f0_method="rmvpe", audio_data, pitch_shift=12)
```

#### Autotune Class
Provides automatic pitch correction functionality.

```python
from rvc.utils import Autotune

ref_freqs = [49.00, 51.91, 55.00, ...]  # Musical note frequencies
autotuner = Autotune(ref_freqs)
adjusted_f0 = autotuner.autotune_f0(original_f0, strength=0.8)
```

## Troubleshooting

### Common Issues

#### Installation Issues
- **Problem**: Installation fails due to dependency conflicts
- **Solution**: Create a fresh virtual environment and install dependencies separately

- **Problem**: PyTorch installation fails
- **Solution**: Install PyTorch separately using the official PyTorch installation guide

#### Runtime Issues
- **Problem**: Out of memory errors
- **Solution**: Reduce batch size or use CPU instead of GPU

- **Problem**: Audio format not supported
- **Solution**: Convert audio to WAV format using FFmpeg

#### NumPy Compatibility Issues
- **Problem**: `AttributeError: module 'numpy' has no attribute 'rint'`
- **Solution**: Upgrade to the latest version of this package which uses `np.round()` instead

### Performance Tips
- Use GPU acceleration when available for faster processing
- Process audio in smaller chunks for large files
- Choose appropriate pitch extraction methods based on quality/speed requirements:
  - Fastest: PM, DIO
  - Balanced: Harvest, SWIPE
  - Highest Quality: RMVPE, CREPE (larger models)

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