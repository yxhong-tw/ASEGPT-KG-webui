import json
import os
from typing import List, Literal, Tuple

from dotenv import load_dotenv
from llama_index.core import Document, KnowledgeGraphIndex, ServiceContext, VectorStoreIndex, get_response_synthesizer, load_index_from_storage
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.core.graph_stores.types import GraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KGTableRetriever, VectorIndexRetriever
from llama_index.core import StorageContext
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from .my_retriever import CustomLLM, CustomRetriever
from app.utils.data_model import KnwoledgeGraphQueryRequest

STORE_EDGE_TYPES, STORE_REL_PROP_NAMES = ['relationship'], ['relationship']
STORE_TAGS = ['entity']


class RagIndexServer:

    def __init__(self, config):
        self.config = config

    def run(self):
        self.load_documents()

        with open(f'{os.getcwd()}/app/data/{self.config["dataset"]["articles_path"]}'
                  ) as f:
            TRIPLETS_LABELED_DATA = json.load(f)

        self.load_store()

    def load_documents(self) -> List[Document]:
        documents = []
        with open(
                f'{os.getcwd()}/app/data/{self.config["dataset"]["documents_path"]}'
        ) as f:
            json_data = json.load(f)

            for doc in json_data:
                # if doc['source_name'] in ['Ptt', '高雄市Open1999']:
                #     continue
                # if '半導體' not in doc['article_content']:
                #     continue

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
                documents.append(document)

        return documents

    def load_store(self) -> Tuple[NebulaGraphStore, bool]:
        # connect to nebula graph
        nebula_config = Config()
        nebula_config.max_connection_pool_size = 10
        nebula_connection_pool = ConnectionPool()
        ok = nebula_connection_pool.init([('graphd', 9669)], nebula_config)

        graph_store = NebulaGraphStore(
            space_name=self.config['storage']['space_name'],
            edge_types=STORE_EDGE_TYPES,
            rel_prop_names=STORE_REL_PROP_NAMES,
            tags=STORE_TAGS)

        return graph_store, ok

    def load_service_and_storage(
            self, store: GraphStore,
            persist_dir: str) -> Tuple[ServiceContext, StorageContext]:
        llm = OpenAI(temperature=0.1, model='gpt-3.5-turbo')
        # llm = LLMPredictor(llm=CustomLLM())
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       chunk_size=4096)

        storage_context = StorageContext.from_defaults(persist_dir=persist_dir,
                                                       graph_store=store)

        return service_context, storage_context
