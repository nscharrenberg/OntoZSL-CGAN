from dp_cgans.utils import Config


# Function is based of: https://stackoverflow.com/a/25294767
def tuplify(data):
    if isinstance(data, list):
        return tuple(map(tuplify, data))

    if isinstance(data, dict):
        return {k: tuplify(v) for k, v in data.items()}

    return data


def load_config(config: str or Config):
    if isinstance(config, str):
        _config = Config(config)
    elif isinstance(config, Config):
        _config = config
    else:
        raise Exception("Configuration could not be read.")

    return _config
