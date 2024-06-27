from Diffusion.pipeline import PromtToImgPipeline
from MicroDreamer.pipeline import ImgTo3dPipeline
import os
import torch
import gc
import random
import string

def promt_to_3D(promt, filename=None) -> str:
    promt_to_img = PromtToImgPipeline()
    #image, variants = promt_to_img(promt)
    image = promt_to_img(promt)

    filepath = 'data/images/'

    if filename is None:
        filename = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(8))
        print(f'Filename generated: {filename}')
    image.save(filepath + f'{filename}.png')
    #for i in range(len(variants)):
    #   variants[i].save(filepath + 'img_' + str(i) + '.png')

    del promt_to_img
    torch.cuda.empty_cache()
    gc.collect()

    img_to_3d = ImgTo3dPipeline()

    os.chdir('MicroDreamer')

    img_to_3d(f'data/images/{filename}.png', 512)
    
    return f'data/3d_models/{filename}.obj'