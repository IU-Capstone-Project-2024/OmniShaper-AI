import torch
from diffusers import StableDiffusion3Pipeline


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

def pipe(n, prompt, negative_prompt="", num_inference_steps=30, guidance_scale=8):
    images = []
    for i in range(n):
        images.append(prompt_to_img(prompt, negative_prompt, num_inference_steps, guidance_scale))
    return images