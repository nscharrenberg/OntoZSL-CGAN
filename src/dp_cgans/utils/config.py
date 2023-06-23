import json
import os.path

from src.dp_cgans.utils.logging import log, LogLevel


class Config:
    """
    A class for loading and accessing configuration data from a JSON file.
    """

    def __init__(self, path: str):
        """
        Initializes a new instance of the Config class.

        Args:
            path (str): The path to the JSON file containing the configuration data.
        """
        if not os.path.isfile(path):
            log(text=f"The expected path \"{path}\" does not seem to be valid.", level=LogLevel.ERROR)

        with open(path, "r") as config_file:
            self.config: dict = json.load(config_file)

    def get(self, key: str, data: dict = None):
        """
        Retrieves the value associated with the specified key from the configuration data.

        Args:
            key (str): The key to look up in the configuration data.
            data (dict, optional): An optional dictionary to use as the configuration data.
                If provided, the lookup is performed on this dictionary instead of the default configuration data.

        Returns:
            The value associated with the specified key, or None if the key is not found in the configuration data.
        """
        current_config: dict = self.config

        if data is not None:
            current_config: dict = data

        if key not in current_config:
            return None

        return current_config[key]

    def get_nested(self, *keys: str):
        """
        Retrieves a nested value from the configuration data using a series of keys.

        Args:
            keys (str): One or more keys representing the path to the nested value in the configuration data.

        Returns:
            The nested value associated with the specified keys, or None if any of the keys are not found in the configuration data.
        """
        if len(keys) <= 0:
            log(text=f"No keys to retrieve from configuration", level=LogLevel.ERROR)

        current_config:dict = self.config

        for key in keys:
            current_config = self.get(key, current_config)

        return current_config
