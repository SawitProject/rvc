<div align="center">

# Simple RVC

A simple, high-quality voice conversion tool focused on simplicity and ease of use.


</div>


# Key Feature

* Pitch extraction methods 20+ like: pm, dio, crepe, fcpe, rmvpe, harvest, yin, pyin and more!
* Powerful command-line interface for batch processing
* Supports multiple embedder models (contentvec, hubert)
* Advanced features: formant shifting, noise reduction, autotune

## Quick Start

### Install

```
pip install git+https://github.com/SawitProject/rvc.git
```

### CLI
```
rvc -i input.wav -o output.wav -m model.pth -p 12
```


## INSTALL

**Requires Python 3.10-3.12 and FFmpeg installed**

### Option 1: Install from Git (Recommended)
```
pip install git+https://github.com/SawitProject/rvc.git
```

### Option 2: Install from Source (Development)
```
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

```
rvc -i input.wav -o output.wav -m model.pth
```

**Basic Options:**
- `-i, --input` - Path to input audio file or directory
- `-o, --output` - Path to output audio file (default: ./output.wav)
- `-m, --model` - Path to .pth model file (required)

**Common Options:**
- `-p, --pitch` - Pitch shift in semitones (default: 0)
- `-f0, --f0_method` - F0 prediction method (default: rmvpe)
- `-idx, --index` - Path to .index file
- `-ir, --index_rate` - Index rate for feature retrieval (default: 0.5)

**Advanced Options:**
- `-split, --split_audio` - Split audio into chunks for processing
- `-clean, --clean_audio` - Apply noise reduction to output
- `-fa, --f0_autotune` - Enable F0 autotune
- `-fs, --formant_shifting` - Enable formant shifting

**Example Usage:**
```
# Simple conversion
rvc -i input.wav -o output.wav -m model.pth -p 12

# Batch conversion from directory
rvc -i ./audio_folder -m model.pth -p 12 -f0 rmvpe

# With index file and autotune
rvc -i input.wav -m model.pth -idx model.index -ir 0.75 -fa
```

For more options, run:
```
rvc --help
```

## Model


**Using CLI:**
- Specify the model path with the `-m` option
- Specify the index path with the `-idx` option (recommended)


**Note:** Pre-trained RVC models can be downloaded from various sources. Ensure you have the right to use any model before converting audio with it.

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
