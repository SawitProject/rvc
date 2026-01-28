# RVC Package Fixes for PyPI/Git Clone Support

## Summary

This document describes the fixes made to the RVC repository to enable proper installation via `pip install git+https://github.com/SawitProject/rvc.git`.

## Problems Fixed

### 1. Removed sys.path Manipulation

**Issue**: The code used `sys.path.append(os.getcwd())` or `sys.path.insert(0, ...)` to add directories to the Python path. This is a hacky workaround that only works when running directly from the source directory and breaks when the package is installed via pip.

**Files Fixed**:
- `gui.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/infer/cli.py` - Removed `sys.path.insert(0, ...)` 
- `rvc/infer/infer.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/infer/pipeline.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/config.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/utils.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/algorithm/attentions.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/algorithm/encoders.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/algorithm/modules.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/algorithm/residuals.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/algorithm/synthesizers.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/generators/hifigan.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/generators/nsf_hifigan.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/generators/refinegan.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/predictor/generator.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/predictor/rmvpe.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/predictor/torchfcpe.py` - Removed `sys.path.append(os.getcwd())`
- `rvc/lib/onnx/onnx_export.py` - Removed `sys.path.append(os.getcwd())` and fixed typo (`loghing` → `logging`)

### 2. Fixed Incorrect Import

**Issue**: In `rvc/infer/cli.py`, line 400 tried to import from `rvc_cli` which doesn't exist.

**Fix**: Changed from `from rvc_cli import __version__` to `from rvc.infer import __version__`

### 3. Added Comments

Added comments to clarify that imports will work when the package is installed via pip:
```python
# Import RVC modules - these will work when the package is installed via pip
```

## Installation Methods

### Method 1: Install from Git (Recommended for Development)

```bash
pip install git+https://github.com/SawitProject/rvc.git
```

### Method 2: Install from Local Directory

```bash
cd /path/to/rvc
pip install .
```

### Method 3: Install in Editable Mode (Development)

```bash
cd /path/to/rvc
pip install -e .
```

## Testing the Installation

After installation, you can verify it works:

```python
import rvc
from rvc.lib.config import Config
from rvc.infer.cli import main

print(f"RVC version: {rvc.infer.__version__}")
```

## Command Line Usage

After installation, the `rvc` command will be available:

```bash
rvc -i input.wav -o output.wav -m model.pth
```

## Package Structure

The package is correctly structured with:
- `pyproject.toml` - Modern Python package configuration
- `requirements.txt` - Dependencies listed
- `rvc/` - Main package directory with proper `__init__.py` files
- All submodules properly organized with their own `__init__.py` files

## Dependencies

The package requires:
- Python 3.10 - 3.12
- PyTorch 2.3.1+
- Various audio processing libraries (librosa, soundfile, etc.)
- ML frameworks (transformers, einops, etc.)

All dependencies are listed in `pyproject.toml` and will be automatically installed when installing the package.

## Notes

1. **CUDA Support**: The package automatically selects GPU or CPU based on availability
2. **Pre-trained Models**: Large model files (>100MB) are not included in the package and should be downloaded separately
3. **DJCM Module**: The `rvc.lib.predictor.djcm` module has a broken import from an external project structure (`from main.library.predictors.DJCM...`). This may need to be fixed separately if DJCM support is required. The rest of the package functions correctly without it.
3. **Platform-specific Dependencies**: The package handles platform-specific dependencies (onnxruntime vs onnxruntime-gpu) automatically
