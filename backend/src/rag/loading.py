from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    load_index_from_storage,
)
from llama_index.graph_stores import NebulaGraphStore
from llama_index.llms import OpenAI
from llama_index.storage.storage_context import StorageContext

from model import CustomLLM

STORE_SPACE_NAME = 'asegpt_rag'
STORE_EDGE_TYPES, STORE_REL_PROP_NAMES = ['relationship'], ['relationship']
STORE_TAGS = ['entity']


def load_store():
    graph_store = NebulaGraphStore(
        space_name=STORE_SPACE_NAME,
        edge_types=STORE_EDGE_TYPES,
        rel_prop_names=STORE_REL_PROP_NAMES,
        tags=STORE_TAGS,
    )

    return graph_store


def load_service_and_storage(store):
    # llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
    llm = LLMPredictor(llm=CustomLLM())
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage_graph", graph_store=store)

    return service_context, storage_context


def load_index(storage_context: StorageContext,
               service_context: ServiceContext):

    # def extract_triplets(input_text):
    #     triplets = triplets_label_df.loc[
    #         triplets_label_df['article_content'] ==
    #         input_text['article_content']]['prediction'][0]

    #     return triplets

    index = load_index_from_storage(
        storage_context=storage_context,
        service_context=service_context,
        max_triplets_per_chunk=100,
        # kg_triplet_extract_fn=extract_triplets,
        space_name=STORE_SPACE_NAME,
        edge_types=STORE_EDGE_TYPES,
        rel_prop_names=STORE_REL_PROP_NAMES,
        tags=STORE_TAGS,
        verbose=True,
    )

    return index
