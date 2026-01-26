"""
RVC CLI Tool - A command-line interface for RVC voice conversion
"""

__version__ = "1.0.0"
__author__ = "BF667"
__email__ = ""

from .cli import main, convert_audio, VoiceConverter
from .infer import run_inference_script 

__all__ = ['main', 'convert_audio', 'VoiceConverter', 'run_inference_script']



