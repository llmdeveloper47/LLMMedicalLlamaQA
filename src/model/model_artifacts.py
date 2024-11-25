from trl import setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import torch

class ModelArtifactLoader:
    def __init__(self, model_path: str, model_type : str):
        self.model_path = model_path
        self.model_type = model_type

    def load_artifacts(self):
        """
        Load and return the model and tokenizer.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: The loaded model and tokenizer.
        """

        if self.model_type == "base":

            base_tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    return_dict=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    device_map="cuda",
                    trust_remote_code=True,
            )
            base_tokenizer.chat_template = None
            base_model, base_tokenizer = setup_chat_format(base_model, base_tokenizer)
            return base_model, base_tokenizer
        
        else:

            finetuned_model = AutoModelForCausalLM.from_pretrained(self.model_path, return_dict=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    device_map="cuda",
                    trust_remote_code=True,)
            
            finetuned_tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            return finetuned_model, finetuned_tokenizer
    


    