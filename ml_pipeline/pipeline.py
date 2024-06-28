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

    current_dir = os.getcwd()

    # Define the target directory
    target_dir = 'MicroDreamer'

    # Change the directory only if the current directory is not the target directory
    if os.path.basename(current_dir) != target_dir:
        os.chdir(target_dir)

    img_to_3d(f'data/images/{filename}.png', 512)

    current_dir = os.getcwd()
    if os.path.basename(current_dir) == target_dir:
        os.chdir('..')
    print(os.path.basename(os.getcwd()))
    return f'data/3d_models/{filename}.obj'