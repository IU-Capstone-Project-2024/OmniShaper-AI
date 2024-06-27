from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
from typing import List

class VariationalDiffusers:
    def __init__(self) -> None:
        """
        Initialisation of the variational diffusers model.
        """
        # an open-source model from a Hugging Face
        self.model_name = "lambdalabs/sd-image-variations-diffusers"

        # loading model into (graphic) memory
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(self.model_name, revision="v2.0")
        self.pipe = self.pipe.to("cuda:0")
    
    def __call__(self, base_image: Image.Image, num_images=3) -> List[Image.Image]:
        """
        Calling an object of a class will result in getting pictures based on test_picture.
        """
        # initialize transformations for test picture
        tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
                ),
            transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
        ])

        # transform test picture and load to graphic memory
        inp = tform(base_image).to("cuda:0").unsqueeze(0)

        # making a list of pictures based on a test one
        images = []
        for _ in range(num_images):
            out = self.pipe(inp, guidance_scale=3)
            images.append(out["images"][0])

        return images