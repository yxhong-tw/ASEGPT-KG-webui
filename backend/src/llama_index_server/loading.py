import json
import os
from typing import List, Literal, Union

import torch
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    PromptTemplate,
    ServiceContext,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import KGTableRetriever, VectorIndexRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.vllm import Vllm
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from .constants import (
    NEBULA_EDGE_TYPES,
    NEBULA_REL_PROP_NAMES,
    NEBULA_STORE_TAGS,
    SYSTEM_PROMPT_SEMICONDUCTOR_ROLE,
)
from .model_v2 import CustomLLM, CustomRetriever

load_dotenv()


def connect_to_nebula_graph() -> bool:
    # connect to nebula graph
    config = Config()
    config.max_connection_pool_size = 10
    connection_pool = ConnectionPool()
    ok = connection_pool.init([('graphd', 9669)], config)

    return ok


def load_store(space_name: str) -> NebulaGraphStore:
    graph_store = NebulaGraphStore(
        space_name=space_name,
        edge_types=NEBULA_EDGE_TYPES,
        rel_prop_names=NEBULA_REL_PROP_NAMES,
        tags=NEBULA_STORE_TAGS,
    )

    return graph_store


def set_global_service(chunk_size: int = 4096,
                       using_openai_gpt: bool = False,
                       local_model_path: str = None,
                       local_tokenizer_path: str = None) -> ServiceContext:

    if using_openai_gpt:
        llm = OpenAI(temperature=0.1, model='gpt-3.5-turbo')
    else:
        # llm = CustomLLM(model_path=local_model_path,
        #                 tokenizer_path=local_tokenizer_path,
        #                 max_new_tokens=4096)

        # llm = Vllm(
        #     model=local_model_path,
        #     tensor_parallel_size=2,
        #     dtype='auto',
        #     # max_new_tokens=32,
        #     temperature=0.3,
        #     top_p=0.8,
        #     top_k=30,
        #     frequency_penalty=0.3,
        #     vllm_kwargs={
        #         # 'swap_space': 1,
        #         "gpu_memory_utilization": 0.95,
        #         'max_model_len': 4096,
        #     },
        # )

        SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
        - Generate human readable output, avoid creating output with gibberish text.
        - Generate only the requested output, don't include any other language before or after the requested output.
        - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
        - Generate professional language typically used in business documents in North America.
        - Never generate offensive or foul language.
        """
        SYSTEM_PROMPT += '\n' + SYSTEM_PROMPT_SEMICONDUCTOR_ROLE

        query_wrapper_prompt = PromptTemplate(
            "<|im_start|>system\n" + SYSTEM_PROMPT +
            "\n\n<|im_end|>\n<|im_start|>user\n{query_str}<|im_end|>\n<|im_start|>assistant\n"
        )

        llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=2048,
            generate_kwargs={
                "temperature": 0.0,
                "do_sample": False
            },
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=local_model_path,
            model_name=local_model_path,
            device_map="auto",
            # change these settings below depending on your GPU
            model_kwargs={
                "torch_dtype": torch.float16,
                "load_in_4bit": True
            },
        )

    embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-m3')
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=chunk_size,
        system_prompt=SYSTEM_PROMPT_SEMICONDUCTOR_ROLE)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return service_context


def load_storage(persist_dir: str, store: GraphStore) -> StorageContext:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir,
                                                   graph_store=store)
    return storage_context


def load_index(documents: List[Document], storage_context: StorageContext,
               service_context: ServiceContext, space_name: str):
    try:
        index = load_index_from_storage(documents=documents,
                                        storage_context=storage_context,
                                        service_context=service_context,
                                        max_triplets_per_chunk=100,
                                        space_name=space_name,
                                        edge_types=NEBULA_EDGE_TYPES,
                                        rel_prop_names=NEBULA_REL_PROP_NAMES,
                                        tags=NEBULA_STORE_TAGS,
                                        verbose=True,
                                        include_embeddings=True,
                                        show_progress=True)
    except Exception:
        print('load index failed')
        return None

    return index


def load_multi_doc_index(documents: List[List[Document]],
                         storage_contexts: List[StorageContext],
                         space_names: List[str]) -> List[BaseIndex]:
    indexes = []
    for i, context in enumerate(storage_contexts):
        try:
            index = load_index_from_storage(
                documents=documents[i],
                storage_context=context,
                max_triplets_per_chunk=100,
                space_name=space_names[i],
                edge_types=NEBULA_EDGE_TYPES,
                rel_prop_names=NEBULA_REL_PROP_NAMES,
                tags=NEBULA_STORE_TAGS,
                verbose=True,
                include_embeddings=True,
                show_progress=True)
        except Exception:
            print(f'load {context.__class__.__name__} failed')

        print(f'Loaded {context.__class__.__name__} successfully!')

        indexes.append(index)

    return indexes


def load_engine(kg_index: KnowledgeGraphIndex,
                mode: Literal['simple', 'hybrid', 'vector',
                              'custom'] = 'simple',
                documents: List[Document] = None,
                service_context: ServiceContext = None) -> BaseQueryEngine:
    if mode == 'simple':
        return kg_index.as_query_engine()
    elif mode == 'hybrid':
        return kg_index.as_query_engine(
            include_text=True,
            response_mode='tree_summarize',
            embedding_mode='hybrid',
            similarity_top_k=5,
            explore_global_knowledge=True,
        )
    elif mode == 'vector':
        vector_index = VectorStoreIndex.from_documents(documents)
        return vector_index.as_query_engine()
    elif mode == 'custom':
        vector_index = VectorStoreIndex.from_documents(documents)
        vector_retriever = VectorIndexRetriever(index=vector_index,
                                                similarity_top_k=20)

        kg_retriever = KGTableRetriever(index=kg_index,
                                        retriever_mode='hybrid',
                                        similarity_top_k=20,
                                        graph_store_query_depth=10,
                                        num_chunks_per_query=20,
                                        include_text=True,
                                        use_global_node_triplets=True)

        custom_retriever = CustomRetriever(vector_retriever,
                                           kg_retriever,
                                           mode='OR')

        response_synthesizer = get_response_synthesizer(
            service_context=service_context,
            response_mode='tree_summarize',
        )

        return RetrieverQueryEngine(retriever=custom_retriever,
                                    response_synthesizer=response_synthesizer)


def load_multi_docs(file_names: List[str]) -> List[List[Document]]:
    documents = []
    for file_name in file_names:
        single_aspect_docs = []
        with open(f'{os.getcwd()}/app/data/{file_name}') as f:
            json_data = json.load(f)
            if 'semiconductor' not in file_name:
                if len(json_data) >= 500:
                    json_data = json_data[:500]
                else:
                    json_data = json_data[:-1]

            for doc in json_data:
                if 'semiconductor' in file_name:
                    if doc['source_name'] in ['Ptt', '高雄市Open1999']:
                        continue
                    if '半導體' not in doc['article_content']:
                        continue

                document = Document(text=doc['article_content'],
                                    metadata={
                                        'crawl_datetime':
                                        doc['crawl_datetime'],
                                        'source_name':
                                        doc['source_name'],
                                        'source_category':
                                        doc['source_category'],
                                        'article_url':
                                        doc['article_url'],
                                        'article_title':
                                        doc['article_title'],
                                        'article_author':
                                        doc['article_author'],
                                        'article_creation_date':
                                        doc['article_creation_date'],
                                    },
                                    excluded_llm_metadata_keys=[
                                        'crawl_datetime', 'source_name',
                                        'source_category', 'article_url',
                                        'article_title', 'article_author',
                                        'article_creation_date'
                                    ])
                single_aspect_docs.append(document)

        documents.append(single_aspect_docs)
        print(f'Loaded {file_name} successfully!')

    return documents


def convert_to_query_engine_tool(
    query_engines: Union[BaseQueryEngine, List[BaseQueryEngine]],
    names: Union[str, List[str]], descriptions: Union[str, List[str]]
) -> Union[QueryEngineTool, List[QueryEngineTool]]:

    assert type(query_engines) == type(names) == type(descriptions)

    assert len(query_engines) == len(names) == len(descriptions)

    return [
        QueryEngineTool.from_defaults(query_engine=e, name=n, description=d)
        for e, n, d in zip(query_engines, names, descriptions)
    ]
