from typing import Any
from test_scripts.LLM import LLM
from test_scripts.InitialDiffusion import DiffusionModel
from test_scripts.VariationalDiffusers import VariationalDiffusers
from PIL import Image

class PromtToImgPipeline:
    
    def __init__(self) -> None:
        """
        Initialisation of the elements of pipeline.
        """
        self.llm = LLM()
        self.diff_model = DiffusionModel()
        #self.var_diff = VariationalDiffusers()
    
    def __call__(self, prompt) -> Any:
        """
        Calling an object of a class will result in getting pictures based on input prompt (passed as argument).
        """
        #enhanced_prompt = self.llm(prompt)
        #image = self.diff_model(enhanced_prompt)
        image = self.diff_model(prompt)
        #variants = self.var_diff(image)

        return image#, variants


# testing block
# ставьте filepath вне гит репа, иначе будет PermissionError: [Errno 13] Permission denied: 'img.png'
