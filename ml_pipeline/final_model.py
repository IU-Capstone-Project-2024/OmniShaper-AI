from test_scripts.pipeline import PromtToImgPipeline
from MicroDreamer.pipeline import ImgTo3dPipeline
import os
import torch
import gc
def promt_to_3D(promt) -> str:
    promt_to_img = PromtToImgPipeline()
    #image, variants = promt_to_img(promt)
    image = promt_to_img(promt)

    filepath = 'images/'

    image.save(filepath + 'img.png')
    #for i in range(len(variants)):
    #   variants[i].save(filepath + 'img_' + str(i) + '.png')

    del promt_to_img
    torch.cuda.empty_cache()
    gc.collect()

    img_to_3d = ImgTo3dPipeline()

    os.chdir('MicroDreamer')
    img_to_3d('images/img.png', 512)
