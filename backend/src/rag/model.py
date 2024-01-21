from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_model(model_path: str, load_in_8bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_8bit=load_in_8bit,
                                                 device_map='auto')
    return model, tokenizer


class CustomLLM(LLM):
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    model, tokenizer = load_model(model_name)
    pipeline = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        device="cuda:0",
    )
    temperature = 0.1
    top_p = 0.8
    top_k = 30
    num_beams = 4
    max_new_tokens = 4096
    repetition_penalty = 1.3

    def set_params(self,
                   temperature=0.1,
                   top_p=0.8,
                   top_k=30,
                   num_beams=4,
                   max_new_tokens=4096):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_len = len(prompt)
        response = self.pipeline(prompt,
                                 max_new_tokens=self.max_new_tokens,
                                 repetition_penalty=self.repetition_penalty,
                                 temperature=self.temperature,
                                 top_p=self.top_p,
                                 top_k=self.top_k,
                                 num_beams=self.num_beams,
                                 stop_sequence=['.'])[0]['generated_text']
        return response[prompt_len:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"
