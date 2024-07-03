import torch
from diffusers import StableDiffusion3Pipeline
import os


def generate_images(prompt, n, request_id, negative_prompt="", num_inference_steps=30, guidance_scale=8):
    images = []

    # create an output folder
    try:
        os.mkdir(f"../data/images/{request_id}")
    except FileExistsError:
        pass

    # init pipe
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    for i in range(n):
        torch.cuda.empty_cache()
        # generate an image and append to array
        images.append(
            pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=512,
                width=512,
                guidance_scale=guidance_scale,
            ).images[0]
        )
        images[-1].save(f"../data/images/{request_id}/{i}.png")

    # clear cuda memory by deleting a pipe
    del pipe

    return images


generate_images('Dinosaur in sweater on white background', 3, 123)
