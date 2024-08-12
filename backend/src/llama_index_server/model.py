from typing import Any, List, Mapping, Optional

import torch
from langchain.llms.base import LLM
from llama_index.core import QueryBundle
from llama_index.core.retrievers import (
    BaseRetriever,
    KGTableRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer,
    Pipeline,
    pipeline,
)


def load_model(model_name_or_path: str,
               tokenizer_name_or_path: str,
               load_in_8bit: bool = False,
               load_in_4bit: bool = False):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name_or_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, quantization_config=quantization_config)
    model = model.bfloat16()

    return model, tokenizer


class CustomLLM(LLM):
    model_name: str = None
    model_folder_path: str = None

    model: AutoModelForCausalLM = None
    tokenizer: LlamaTokenizer = None
    generation_pipeline: Pipeline = None

    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 30
    num_beams: int = 4
    max_new_tokens: int = 4096
    repetition_penalty: float = 1.3

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 temperature=0.1,
                 top_p=0.8,
                 top_k=30,
                 num_beams=4,
                 repetition_penalty=1.3,
                 max_new_tokens=4096):
        super(CustomLLM, self).__init__()

        self.model, self.tokenizer = load_model(model_path,
                                                tokenizer_path,
                                                load_in_4bit=True,
                                                load_in_8bit=False)
        self.generation_pipeline = pipeline('text-generation',
                                            model=self.model,
                                            tokenizer=self.tokenizer,
                                            device_map='auto')

        self.model_name = model_path
        self.model_folder_path = model_path

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              **kwargs) -> str:
        params = {**self._get_model_default_parameters, **kwargs}
        response = self.generation_pipeline(prompt,
                                            **params)[0]['generated_text']
        return response

    @property
    def _get_model_default_parameters(self):
        return {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'num_beams': self.num_beams,
            'max_new_tokens': self.max_new_tokens,
            'repetition_penalty': self.repetition_penalty,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""

        return {
            'model_name': self.model_name,
            'model_path': self.model_folder_path,
            'model_parameters': self._get_model_default_parameters
        }

    @property
    def _llm_type(self) -> str:
        return 'custom'


class CustomRetriever(BaseRetriever):

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = 'OR',
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ('AND', 'OR'):
            raise ValueError('Invalid mode.')
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        query_bundle.query_str = query_bundle.query_str[:510]
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == 'AND':
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
