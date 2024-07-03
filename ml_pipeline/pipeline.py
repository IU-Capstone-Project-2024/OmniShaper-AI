import torch
from diffusers import StableDiffusion3Pipeline
from MicroDreamer.pipeline import ImgTo3dPipeline
import os
import torch
import gc
import random
import string

def prompt_to_img(prompt, negative_prompt="", num_inference_steps=30, guidance_scale=8):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        height=512,
        width=512,
        guidance_scale=guidance_scale,
    ).images[0]

    del pipe
    return image

def promt_to_3D(promt, filename=None) -> str:

    image = prompt_to_img(promt)

    filepath = 'data/images/'

    if filename is None:
        filename = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(8))
        print(f'Filename generated: {filename}')
    image.save(filepath + f'{filename}.png')
    #for i in range(len(variants)):
    #   variants[i].save(filepath + 'img_' + str(i) + '.png')

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