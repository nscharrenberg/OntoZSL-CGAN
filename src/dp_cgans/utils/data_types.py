from dp_cgans.utils import Config
from dp_cgans.utils.logging import log, LogLevel


def tuplify(data):
    """
    Convert a nested data structure (list or dictionary) into a tuple-based representation.

    Args:
        data (list or dict): The input data structure to be converted into a tuple-based representation.

    Returns:
        tuple: The tuple-based representation of the input data structure.

    Note:
        This implementation is based on the solution provided on Stack Overflow:
        https://stackoverflow.com/a/25294767
    """
    if isinstance(data, list):
        return tuple(map(tuplify, data))

    if isinstance(data, dict):
        return {k: tuplify(v) for k, v in data.items()}

    return data


def load_config(config: str or Config) -> Config:
    """
    Load a configuration object from either a string or an existing Config object.

    Args:
        config (str or Config): The configuration input, which can be a string or an instance of the Config class.

    Returns:
        Config: The loaded Config object.

    Note:
        This function assumes the existence of the Config class and a logging mechanism for error messages.
        The details of the Config class and logging mechanism should be provided elsewhere in the codebase.
    """
    if isinstance(config, str):
        _config = Config(config)
    elif isinstance(config, Config):
        _config = config
    else:
        log(text="Configuration could not be read.", level=LogLevel.ERROR)

    return _config
