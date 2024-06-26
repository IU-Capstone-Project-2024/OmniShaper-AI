from typing import Any
from LLM import LLM
from InitialDiffusion import DiffusionModel
from VariationalDiffusers import VariationalDiffusers
from PIL import Image

class PromtToImgPipeline:
    
    def __init__(self) -> None:
        """
        Initialisation of the elements of pipeline.
        """
        self.llm = LLM()
        self.diff_model = DiffusionModel()
        self.var_diff = VariationalDiffusers()
    
    def __call__(self, prompt) -> Any:
        """
        Calling an object of a class will result in getting pictures based on input prompt (passed as argument).
        """
        enhanced_prompt = self.llm(prompt)
        image = self.diff_model(enhanced_prompt)
        variants = self.var_diff(image)

        return image, variants


# testing block
# ставьте filepath вне гит репа, иначе будет PermissionError: [Errno 13] Permission denied: 'img.png'

pipe =  Pipeline()
image, variants = pipe('house')

filepath = 'images/'

image.save(filepath + 'img.png')
for i in range(len(variants)):
     variants[i].save(filepath + 'img_' + str(i) + '.png')

