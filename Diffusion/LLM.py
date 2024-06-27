from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLM:
    def __init__(self) -> None:
        """
        Initialisation of the LLM.
        """
        # an open-source model from a Hugging Face
        self.model_name = "vahn9995/longt5-stable-diffusion-prompt"

        # loading tokenizer and model into memory
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def __call__(self, prompt:str) -> str:
        """
        Calling an object of a class with an argument will result in getting enhansed prompt.
        """
        # tokenize initial prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        # generate new (tokenized) prompt
        output_ids = self.model.generate(input_ids)
        # decode tokens into string
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # just for logging
        # TODO: remove after testing
        print("Initial prompt:", prompt)
        print("Enhansed prompt:", output_text)

        return output_text