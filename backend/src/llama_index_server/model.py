import os
from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever, KGTableRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from transformers import AutoModelForCausalLM, LlamaTokenizer, pipeline


def load_model(model_path: str,
               load_in_8bit: bool = False,
               load_in_4bit: bool = False):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=load_in_8bit,
        # load_in_4bit=load_in_4bit,
        device_map='auto')
    return model, tokenizer


class CustomLLM(LLM):

    def __init__(self):

        model_name = f'{os.getcwd()}/app/models/triplet_rationale_chatml_qlora/merged'
        self.model, self.tokenizer = load_model(model_name, load_in_8bit=True)
        self.pipeline = pipeline(
            task='text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.temperature = 0.1
        self.top_p = 0.8
        self.top_k = 30
        self.num_beams = 4
        self.max_new_tokens = 4096
        self.repetition_penalty = 1.3

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
        return {'name_of_model': self.model_name}

    @property
    def _llm_type(self) -> str:
        return 'custom'


class CustomRetriever(BaseRetriever):

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
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
