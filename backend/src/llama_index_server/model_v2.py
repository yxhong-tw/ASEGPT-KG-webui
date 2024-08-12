import json
import os
from typing import Any, Dict, List, Mapping, Optional, Union

import openai
from langchain.llms.base import LLM
from llama_index.core import QueryBundle
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.retrievers import (
    BaseRetriever,
    KGTableRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore
from llama_index.llms.vllm import Vllm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from vllm import LLM, SamplingParams

from .constants import DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPT_SEMICONDUCTOR_ROLE

openai.OpenAI.api_key = os.getenv('OPENAI_API_KEY')


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(client, **kwargs) -> openai.ChatCompletion:
    return client.chat.completions.create(**kwargs)


def call_gpt_api(client: openai.OpenAI,
                 prompt: str,
                 model='gpt-3.5-turbo',
                 max_tokens=1024,
                 system_prompt=DEFAULT_SYSTEM_PROMPT) -> str:

    messages = [{
        'role': 'system',
        'content': system_prompt
    }, {
        'role': 'user',
        'content': prompt
    }]
    completion = chatcompletion_with_backoff(client,
                                             model=model,
                                             max_tokens=max_tokens,
                                             temperature=0,
                                             messages=messages)

    res = completion.choices[0].message.content

    return res


def load_model(model_name_or_path: str):
    # model = LLM(
    #     model=model_name_or_path,
    #     max_model_len=8192,
    #     # max_model_len=4096,
    #     tensor_parallel_size=2,
    #     # gpu_memory_utilization=0.8,
    #     quantization='awq'
    #     # dtype='half',
    #     # max_num_seqs=64
    # )

    model = Vllm(
        model=model_name_or_path,
        tensor_parallel_size=2,
        dtype='float16',
        # max_new_tokens=32,
        temperature=0.3,
        top_p=0.8,
        top_k=30,
        frequency_penalty=0.3,
        # use_beam_search=True,
        vllm_kwargs={
            'swap_space': 1,
            "gpu_memory_utilization": 0.9,
            # 'max_model_len': 4096,
            # 'num_beams': 5,
        },
    )

    return model


class CustomLLM(LLM):
    model_name: str = None
    model_folder_path: str = None

    model: Union[openai.OpenAI, LLM] = None

    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 30
    num_beams: int = 4
    max_new_tokens: int = 4096
    repetition_penalty: float = 1.3

    system_prompt: str = None
    query_wrapper_prompt = None
    pydantic_program_mode = None
    metadata: LLMMetadata = LLMMetadata()

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 temperature=0.1,
                 top_p=0.8,
                 top_k=30,
                 num_beams=4,
                 repetition_penalty=1.3,
                 max_new_tokens=4096):
        # super(CustomLLM, self).__init__()

        call_gpt_api = False
        if not call_gpt_api:
            self.model = load_model(model_path)
        else:
            self.model = openai.OpenAI()

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

        print('[Main Task]')
        print(f'Q: {prompt}')

        # sampling_params = SamplingParams(max_tokens=params['max_new_tokens'],
        #                                  temperature=params['temperature'],
        #                                  top_k=params['top_k'],
        #                                  top_p=params['top_p'],
        #                                  frequency_penalty=0.2)
        # outputs = self.model.generate(prompt, sampling_params)
        # response = [output.outputs[0].text for output in outputs][0]
        response = self.model.complete([prompt]).text

        # response = call_gpt_api(self.model,
        #                         prompt,
        #                         max_tokens=params['max_new_tokens'])

        print(f'A: {response}\n')

        return response

    def predict(self, query_str: str, **kwargs) -> str:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        task_name = ''

        if kwargs.get('question', None) is not None and kwargs.get(
                'max_keywords', None) is not None:
            prompt_template = query_str.template
            question = kwargs['question']
            max_keywords = kwargs.get('max_keywords')
            query_str = prompt_template.format(question=question,
                                               max_keywords=max_keywords)
            task_name = 'keywords_generation'
        elif kwargs.get('context_str', None) is not None:
            system_prompt = SYSTEM_PROMPT_SEMICONDUCTOR_ROLE
            prompt_template = query_str.default_template.template
            question = query_str.default_template.kwargs['query_str']
            context_str = kwargs['context_str']
            query_str = prompt_template.format(context_str=context_str,
                                               query_str=question)
            task_name = 'response_synthesis'
        else:
            system_prompt = f"Let's think step by step. Please provide a longer and more detailed answer and Use the chinese to answer the question. {SYSTEM_PROMPT_SEMICONDUCTOR_ROLE}"
            task_name = 'others'

        query_str = system_prompt + '\n' + query_str
        print(f'[Intermidate Task: {task_name}]')
        print(f'Q: {query_str}')

        params = self._get_model_default_parameters
        # sampling_params = SamplingParams(max_tokens=params['max_new_tokens'],
        #                                  temperature=params['temperature'],
        #                                  top_k=params['top_k'],
        #                                  top_p=params['top_p'],
        #                                  frequency_penalty=0.2)
        # outputs = self.model.generate(query_str, sampling_params)
        # response = [output.outputs[0].text for output in outputs][0]

        response = self.model.complete([query_str]).text

        # response = call_gpt_api(self.model,
        #                         query_str,
        #                         max_tokens=params['max_new_tokens'],
        #                         system_prompt=system_prompt)

        print(f'A: {response}\n')

        # if task_name == 'keywords_generation':
        #     if '中美貿易戰' in query_str:
        #         response = 'KEYWORDS: 半導體, 中美貿易戰, 貿易, 封裝, 製程, 晶片製造, 進出口, 晶圓廠, 全球半導體局勢, 經濟發展'
        #     elif 'AI' in query_str:
        #         response = 'KEYWORDS: 半導體, AI, 人工智慧, AI晶片, AI chips, 晶片製造, 全球半導體局勢, 經濟發展'
        #     elif '矽光子' in query_str:
        #         response = 'KEYWORDS: 半導體, 半導體材料, 矽光子, 封裝, 製程, 晶片製造, 光刻, 晶圓廠, 全球半導體局勢, 經濟發展'

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
        mode: str = "OR",
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

        if len(query_bundle.query_str) > 510:
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
