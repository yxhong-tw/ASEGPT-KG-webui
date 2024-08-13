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

# WAR_DOCS, SILICON_DOCS, AI_DOCS = load_multi_docs([
#     'processed_中美貿易戰禁令_202306-202401.json',
#     'processed_矽光子發展_202306-202401.json',
#     'processed_AI晶片發展_0-500_202306-202401.json'
# ])

# is_connected = connect_to_nebula_graph()
# print(f'NebulaGraph connected: {is_connected}')

# print('Start loading graph store')
# WAR_STORE = load_store('asegptkg_rag_war')
# print('War Graph store loaded successfully!')
# SILICON_STORE = load_store('asegptkg_rag_silicon')
# print('Silicon Graph store loaded successfully!')
# AI_STORE = load_store('asegptkg_rag_ai')
# print('AI Graph store loaded successfully!')

# print('Start loading service and storage')
# SERVICE_CONTEXT = set_global_service(
#     using_openai_gpt=False,
#     chunk_size=4096,
#     local_model_path=f'{os.getcwd()}/app/models/qa_20240401/checkpoint-744/',
#     local_tokenizer_path=f'{os.getcwd()}/app/models/qa_20240401/')
# WAR_STORAGE_CONTEXT = load_storage(
#     store=WAR_STORE,
#     persist_dir=f'{os.getcwd()}/app/storages/war',
# )
# SILICON_STORAGE_CONTEXT = load_storage(
#     store=SILICON_STORE, persist_dir=f'{os.getcwd()}/app/storages/silicon')
# AI_STORAGE_CONTEXT = load_storage(store=AI_STORE,
#                                   persist_dir=f'{os.getcwd()}/app/storages/ai')
# print('Service and storage loaded successfully!')

# print('Start loading index')
# WAR_KG_INDEX, SILICON_KG_INDEX, AI_KG_INDEX = load_multi_doc_index(
#     documents=[WAR_DOCS, SILICON_DOCS, AI_DOCS],
#     storage_contexts=[
#         WAR_STORAGE_CONTEXT, SILICON_STORAGE_CONTEXT, AI_STORAGE_CONTEXT
#     ],
#     space_names=[
#         'asegptkg_rag_war', 'asegptkg_rag_silicon', 'asegptkg_rag_ai'
#     ])
# print('Index loaded successfully!')

# WAR_QUERY_ENGINE = load_engine(WAR_KG_INDEX,
#                                mode='custom',
#                                documents=WAR_DOCS,
#                                service_context=SERVICE_CONTEXT)
# SILICON_QUERY_ENGINE = load_engine(SILICON_KG_INDEX,
#                                    mode='custom',
#                                    documents=SILICON_DOCS,
#                                    service_context=SERVICE_CONTEXT)
# AI_QUERY_ENGINE = load_engine(AI_KG_INDEX,
#                               mode='custom',
#                               documents=AI_DOCS,
#                               service_context=SERVICE_CONTEXT)
# print('Query engine loaded successfully!')

# WAR_QUERY_ENGINE_TOOL = QueryEngineTool.from_defaults(
#     query_engine=WAR_QUERY_ENGINE,
#     name='war_query_engine_tool',
#     description=
#     'Useful for answering questions about "China–United States trade war" and its impact on the world',
# )
# SILICON_QUERY_ENGINE_TOOL = QueryEngineTool.from_defaults(
#     query_engine=SILICON_QUERY_ENGINE,
#     name='silicon_query_engine_tool',
#     description=
#     'Useful for answering questions about "Silicon Photonics development trend" and its impact on the world',
# )
# AI_QUERY_ENGINE_TOOL = QueryEngineTool.from_defaults(
#     query_engine=AI_QUERY_ENGINE,
#     name='ai_query_engine_tool',
#     description=
#     'Useful for answering questions about "Artificial Intelligence(AI)" and "AI Chips development trend" and its impact on the world',
# )

# RAG_QUERY_ENGINE = RouterQueryEngine(selector=LLMMultiSelector.from_defaults(),
#                                      query_engine_tools=[
#                                          WAR_QUERY_ENGINE_TOOL,
#                                          SILICON_QUERY_ENGINE_TOOL,
#                                          AI_QUERY_ENGINE_TOOL
#                                      ],
#                                      service_context=SERVICE_CONTEXT)
# print('router query engine constructed successfully!')


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
        using_openai_gpt=True,
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
    doc_paths=[
        'processed_中美貿易戰禁令_202306-202401.json',
        'processed_矽光子發展_202306-202401.json',
        'processed_AI晶片發展_0-500_202306-202401.json',
        'processed_semiconductor_0-30000_articles_20230901-20230921.json'
    ],
    space_names=[
        'asegptkg_rag_war', 'asegptkg_rag_silicon', 'asegptkg_rag_ai',
        'asegptkg_rag_semiconductor'
    ],
    persist_dirs=[
        f'{os.getcwd()}/app/storages/war',
        f'{os.getcwd()}/app/storages/silicon',
        f'{os.getcwd()}/app/storages/ai',
        f'{os.getcwd()}/app/storages/semiconductor'
    ],
    generator_model_path=
    f'{os.getcwd()}/app/models/rag_it_chatml_qlora_lr6e-4_wd5e-3-trainset_20240608_multi_task_instruction_tuning_no_response_synthesis_rag_cp_freeze-top25layers-lm_head_chat-vector/',
    generator_tokenizer_path=
    f'{os.getcwd()}/app/models/rag_it_chatml_qlora_lr6e-4_wd5e-3-trainset_20240608_multi_task_instruction_tuning_no_response_synthesis_rag_cp_freeze-top25layers-lm_head_chat-vector/'
    # generator_model_path='TheBloke/Mistral-7B-Instruct-v0.2-AWQ',
    # generator_tokenizer_path='TheBloke/Mistral-7B-Instruct-v0.2-AWQ'
    )
WAR_QUERY_ENGINE, SILICON_QUERY_ENGINE, AI_QUERY_ENGINE, SEMICONDUCTOR_QUERY_ENGINE = QUERY_ENGINES


@app.post('/query')
def kg_query(query: KnwoledgeGraphQueryRequest):
    print(f'Query: {query.query}')

    if query.query in app.cache:
        cache_response = app.cache[query.query]
        return cache_response

    response = RAG_QUERY_ENGINE.query(query.query)
    # try:
    #     response = RAG_QUERY_ENGINE.query(query.query)
    # except Exception as e:
    #     print('Query failed')
    #     print(e)

    #     try:
    #         if 'ai' in query.query.lower():
    #             print('Retry with AI_QUERY_ENGINE')
    #             response = AI_QUERY_ENGINE.query(query.query)
    #         elif '矽光子' in query.query.lower():
    #             print('Retry with SILICON_QUERY_ENGINE')
    #             response = SILICON_QUERY_ENGINE.query(query.query)
    #         elif '中美貿易戰' in query.query.lower():
    #             print('Retry with WAR_QUERY_ENGINE')
    #             response = WAR_QUERY_ENGINE.query(query.query)
    #         else:
    #             return {
    #                 'response':
    #                 'Something went wrong, please contact the admin',
    #                 'relevant_nodes': {},
    #                 'highlight': {
    #                     'relation': [],
    #                     'entity': []
    #                 }
    #             }
    #     except Exception as e:
    #         print('Retry failed')
    #         print(e)
    #         return {
    #             'response': 'Something went wrong, please contact the admin',
    #             'relevant_nodes': {},
    #             'highlight': {
    #                 'relation': [],
    #                 'entity': []
    #             }
    #         }
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
