import yaml


def load_config(file_path: str):
    with open(file_path, 'r') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)

    return config_file


def set_rag_default_config(config):
    if 'data' not in config:
        config['data'] = {}
    if 'nebula_graph' not in config:
        config['nebula_graph'] = {}
    if 'rag' not in config:
        config['rag'] = {}

    if 'document_paths' not in config['data']:
        config['data']['document_paths'] = []
    if 'space_names' not in config['nebula_graph']:
        config['nebula_graph']['space_names'] = []
    if 'persist_dirs' not in config['nebula_graph']:
        config['nebula_graph']['persist_dirs'] = []

    if 'generator_model_path' not in config['rag']:
        config['rag']['generator_model_path'] = ''
    if 'generator_tokenizer_path' not in config['rag']:
        config['rag']['generator_tokenizer_path'] = ''
    if 'query_engine_tools' not in config['rag']:
        config['rag']['query_engine_tools'] = {}
    if 'using_openai_gpt' not in config['rag']:
        config['rag']['using_openai_gpt'] = False

    return config
