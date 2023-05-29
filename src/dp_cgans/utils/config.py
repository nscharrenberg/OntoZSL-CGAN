import json
import os.path


class Config:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise Exception(f"The expected path \"{path}\" does not seem to be valid.")

        with open(path, "r") as config_file:
            self.config: dict = json.load(config_file)

    def get(self, key: str, data: dict = None):
        """
        Retrieve the value of a specific first-level key from the configuration dictionary.

        Args:
            key: The key to search
            data: (Optional) The data to search the key in. If now data is passed the internal self.config will be used to search.

        Returns: The value of the key

        """
        current_config: dict = self.config

        if data is not None:
            current_config = data

        if key not in current_config:
            return None

        return current_config[key]

    def get_nested(self, *keys: str):
        """
        Retrieve the value of a any-level key from the configuration dictionary.
        Args:
            *keys: Breadcrumb of all the level keys to go through, the reach the most right key. e.g. 'level1', 'level2', 'level3'.

        Returns: The value of the nested key.

        """
        if len(keys) <= 0:
            raise KeyError(f"No keys to retrieve from configuration")

        current_config: dict = self.config

        for key in keys:
            current_config = self.get(key, current_config)

        return current_config
