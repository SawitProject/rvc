import os
import sys
import torch

sys.path.append(os.getcwd())

from rvc.lib.backend import opencl

PREDICTOR_MODEL = os.path.join(os.getcwd(), "assets", " models")




def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances: instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Config:
    def __init__(self, cpu_mode=False, is_half=False):
        self.device = "cuda:0" if torch.cuda.is_available() else ("ocl:0" if opencl.is_available() else "cpu")
        self.is_half = is_half
        self.gpu_mem = None
        self.cpu_mode = cpu_mode
        if cpu_mode: self.device = "cpu"

    def device_config(self):
        if not self.cpu_mode:
            if self.device.startswith("cuda"): self.set_cuda_config()
            elif opencl.is_available(): self.device = "ocl:0"
            elif self.has_mps(): self.device = "mps"
            else: self.device = "cpu"

        if self.gpu_mem is not None and self.gpu_mem <= 4: return 1, 5, 30, 32
        return (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024**3)

    def has_mps(self):
        return torch.backends.mps.is_available()