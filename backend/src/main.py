import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector, PydanticMultiSelector
from llama_index.core.tools import QueryEngineTool

from .llama_index_server import (
    load_documents,
    load_engine,
    load_index,
    load_multi_doc_index,
    load_multi_docs,
    load_service_and_storage,
    load_storage,
    load_store,
    set_global_service,
)
from .utils.data_model import KnwoledgeGraphQueryRequest
from .utils.data_util import load_config
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

# config = load_config(f'{os.getcwd()}/app/data/config.yaml')
# print(config['dataset'])

WAR_DOCS, SILLICON_DOCS = load_multi_docs([
    'processed_中美貿易戰禁令_202306-202401.json',
    'processed_矽光子發展_202306-202401.json'
])
# SILLICON_DOCS,  = load_multi_docs(['processed_矽光子發展_202306-202401.json'])

print('Start loading graph store')
WAR_STORE, is_connected = load_store('asegptkg_rag_war')
SILLICON_STORE, is_connected = load_store('asegptkg_rag_sillicon')
print(f'Graph store loaded successfully? {is_connected}')

print('Start loading service and storage')
SERVICE_CONTEXT = set_global_service()
WAR_STORAGE_CONTEXT = load_storage(
    store=WAR_STORE,
    persist_dir=f'{os.getcwd()}/app/storages/war',
)
SILLICON_STORAGE_CONTEXT = load_storage(
    store=SILLICON_STORE, persist_dir=f'{os.getcwd()}/app/storages/sillicon')
print('Service and storage loaded successfully!')

print('Start loading index')
WAR_KG_INDEX, SILLICON_KG_INDEX = load_multi_doc_index(
    documents=[WAR_DOCS, SILLICON_DOCS],
    storage_contexts=[WAR_STORAGE_CONTEXT, SILLICON_STORAGE_CONTEXT],
    space_names=['asegptkg_rag_war', 'asegptkg_rag_sillicon'])
# SILLICON_KG_INDEX,  = load_multi_doc_index(
#     documents=[SILLICON_DOCS],
#     storage_contexts=[SILLICON_STORAGE_CONTEXT],
#     space_names=['asegptkg_rag_sillicon'])
print('Index loaded successfully!')

WAR_QUERY_ENGINE = load_engine(WAR_KG_INDEX,
                               mode='custom',
                               documents=WAR_DOCS,
                               service_context=SERVICE_CONTEXT)
SILLICON_QUERY_ENGINE = load_engine(SILLICON_KG_INDEX,
                                    mode='custom',
                                    documents=SILLICON_DOCS,
                                    service_context=SERVICE_CONTEXT)
print('Query engine loaded successfully!')

WAR_QUERY_ENGINE_TOOL = QueryEngineTool.from_defaults(
    query_engine=WAR_QUERY_ENGINE,
    name='war_query_engine_tool',
    description=
    "Useful for answering questions about China–United States trade war and its impact on the world",
)
SILLICON_QUERY_ENGINE_TOOL = QueryEngineTool.from_defaults(
    query_engine=SILLICON_QUERY_ENGINE,
    name='sillicon_query_engine_tool',
    description=
    "Useful for answering questions about this Sillicon development trend and its impact on the world",
)

RAG_QUERY_ENGINE = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[WAR_QUERY_ENGINE_TOOL, SILLICON_QUERY_ENGINE_TOOL],
    service_context=SERVICE_CONTEXT)
print('router query engine constructed successfully!')


@app.post('/query')
def kg_query(query: KnwoledgeGraphQueryRequest):
    print(f'Query: {query.query}')

    if query.query in app.cache:
        cache_response = app.cache[query.query]
        return cache_response

    response = RAG_QUERY_ENGINE.query(query.query)

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
