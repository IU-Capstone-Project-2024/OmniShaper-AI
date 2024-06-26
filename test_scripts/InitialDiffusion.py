from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

class DiffusionModel:
    def __init__(self) -> None:
        """
        Initialisation of the diffusion model.
        """
        # an open-source model from a Hugging Face
        self.model_name = "runwayml/stable-diffusion-v1-5"

        # loading model into (graphic) memory
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.pipe = self.pipe
    
    def __call__(self, prompt: str) -> Image.Image:
        """
        Calling an object of a class with an argument will result in getting a picture based on prompt.
        """
        # negative prompt to make result better
        negative_prompt = "full scale, bad quality, background"
        image = self.pipe(prompt, negative_prompt=negative_prompt).images[0]
        return image