from Diffusion.pipeline import PromtToImgPipeline
from MicroDreamer.pipeline import ImgTo3dPipeline
import os
import torch
import gc
import random
import string

def promt_to_3D(promt, filename=None) -> str:
    
    filename = 'lm5dap0e'
    return f'data/3d_models/{filename}_mesh.obj'