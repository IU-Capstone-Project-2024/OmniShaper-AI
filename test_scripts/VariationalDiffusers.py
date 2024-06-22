from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
from typing import List

class VariationalDiffusers:
    def __init__(self, image: Image.Image) -> None:
        """
        Initialisation of the variational diffusers model.
        """
        # an open-source model from a Hugging Face
        self.model_name = "lambdalabs/sd-image-variations-diffusers"

        # loading model into (graphic) memory
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(self.model_name, revision="v2.0")
        self.pipe = self.pipe.to("cuda:0")

        # getting base image
        self.base_image = image
    
    def __call__(self, num_images=5) -> List[Image.Image]:
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
        inp = tform(self.base_image).to("cuda:0").unsqueeze(0)

        # just for logging
        # TODO: remove after testing
        out = self.pipe(inp, guidance_scale=3)
        out["images"][0].save("result.jpg")

        # making a list of pictures based on a test one
        images = []
        for _ in range(num_images):
            out = self.pipe(inp, guidance_scale=3)
            images.append(out["images"][0])

        return images