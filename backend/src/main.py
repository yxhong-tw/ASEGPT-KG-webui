import os
from typing import List, Tuple

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.query_engine import BaseQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

from .llama_index_server import (
    RAG_QUERY_ENGINE_TOOLS_MAPPING,
    CustomMultiSelector,
    connect_to_nebula_graph,
    convert_to_query_engine_tool,
    load_engine,
    load_multi_doc_index,
    load_multi_docs,
    load_storage,
    load_store,
    set_global_service,
)
from .utils.data_model import KnwoledgeGraphQueryRequest
from .utils.data_util import load_config, set_rag_default_config
from .utils.kg_util import find_triplet_index, parse_triplets

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.cache = {}

config = load_config('./configs/default.yaml')
config = set_rag_default_config(config)


def initialize_rag_settings(
    doc_paths: List[str], space_names: List[str], persist_dirs: List[str],
    generator_model_path: str, generator_tokenizer_path: str
) -> Tuple[RouterQueryEngine, List[BaseQueryEngine]]:
    assert len(space_names) == len(
        persist_dirs
    ), 'Length of space_names and persist_dirs should be the same.'

    print(f'Using GPU: {torch.cuda.is_available()}')

    documents = load_multi_docs(doc_paths)

    service_context = set_global_service(
        using_openai_gpt=False,
        chunk_size=4096,
        local_model_path=generator_model_path,
        local_tokenizer_path=generator_tokenizer_path)
    stores = [load_store(space_name) for space_name in space_names]
    storages = [
        load_storage(store=s, persist_dir=p)
        for s, p in zip(stores, persist_dirs)
    ]

    indices = load_multi_doc_index(documents=documents,
                                   storage_contexts=storages,
                                   space_names=space_names)
    engines = [
        load_engine(i,
                    mode='custom',
                    documents=d,
                    service_context=service_context)
        for i, d in zip(indices, documents)
    ]

    engine_tools = convert_to_query_engine_tool(
        engines,
        names=[
            RAG_QUERY_ENGINE_TOOLS_MAPPING[space_name]['name']
            for space_name in space_names
        ],
        descriptions=[
            RAG_QUERY_ENGINE_TOOLS_MAPPING[space_name]['description']
            for space_name in space_names
        ])

    return RouterQueryEngine(
        # selector=LLMMultiSelector.from_defaults(),
        selector=CustomMultiSelector(),
        query_engine_tools=engine_tools,
        service_context=service_context), engines


RAG_QUERY_ENGINE, QUERY_ENGINES = initialize_rag_settings(
    doc_paths=config['data']['document_paths'],
    space_names=config['nebula_graph']['space_names'],
    persist_dirs=[
        os.path.join(os.getcwd(), path)
        for path in config['nebula_graph']['persist_dirs']
    ],
    generator_model_path=os.path.join(os.getcwd(),
                                      config['rag']['generator_model_path']),
    generator_tokenizer_path=os.path.join(
        os.getcwd(), config['rag']['generator_tokenizer_path']))

WAR_QUERY_ENGINE, SILICON_QUERY_ENGINE, \
    AI_QUERY_ENGINE, SEMICONDUCTOR_QUERY_ENGINE = QUERY_ENGINES


@app.post('/query')
def kg_query(query: KnwoledgeGraphQueryRequest):
    print(f'Query: {query.query}')

    if query.query in app.cache:
        cache_response = app.cache[query.query]
        return cache_response

    response = RAG_QUERY_ENGINE.query(query.query)
    print(f'Response: {response}')

    relevant_nodes = {}
    highlight_relations, highlight_entities = [], []
    for resp_id in response.metadata.keys():
        node = [n for n in response.source_nodes if n.node.id_ == resp_id]
        if not node:
            continue
        node = node[0]

        relevant_nodes[resp_id] = {'score': node.score}

        if node.score != 1000:
            relevant_nodes[resp_id]['content'] = node.node.text
            relevant_nodes[resp_id]['metadata'] = node.node.metadata

        if 'kg_rel_texts' in node.node.metadata:
            relevant_nodes[resp_id]['triplets'] = []

            for one_data in node.node.metadata['kg_rel_texts']:
                triplets = parse_triplets(one_data)
                relevant_nodes[resp_id]['triplets'].extend(triplets)

            # highlight_relations, highlight_entities, _ = find_triplet_index(
            #     relevant_nodes[resp_id]['triplets'], response.response)

    results = {
        'response': response.response,
        'relevant_nodes': relevant_nodes,
        'highlight': {
            'relation': highlight_relations,
            'entity': highlight_entities
        }
    }
    app.cache[query.query] = results

    return results
