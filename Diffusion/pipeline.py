from typing import Any
from Diffusion.InitialDiffusion import DiffusionModel
from Diffusion.VariationalDiffusers import VariationalDiffusers


class PromtToImgPipeline:
    
    def __init__(self) -> None:
        """
        Initialisation of the elements of pipeline.
        """
        # self.llm = LLM()
        self.diff_model = DiffusionModel()
        # self.var_diff = VariationalDiffusers()
    
    def __call__(self, prompt) -> Any:
        """
        Calling an object of a class will result in getting pictures based on input prompt (passed as argument).
        """
        image = self.diff_model(prompt)
        # variants = self.var_diff(image)

        return image
