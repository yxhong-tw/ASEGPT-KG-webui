import yaml


def load_config(file_path: str):
    with open(file_path, 'r') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)

    return config_file
