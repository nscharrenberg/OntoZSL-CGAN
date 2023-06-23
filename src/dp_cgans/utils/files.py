import codecs
import os

import pandas as pd
import requests

from dp_cgans.utils.logging import log, LogLevel


def recode(source: str, target: str, decoding: str = "ANSI", encoding: str = "utf-8", verbose: bool = False) -> None:
    """
    Recodes a file from one encoding to another.

    Args:
        source (str): The path to the source file.
        target (str): The path to the target file where the recoded contents will be saved.
        decoding (str, optional): The encoding of the source file. Defaults to "ANSI".
        encoding (str, optional): The encoding to recode the file into. Defaults to "utf-8".
        verbose (bool, optional): If True, displays log messages. Defaults to False.

    Returns:
        None
    """
    log(text=f"Preparing file to be recoded from {decoding} to {encoding}...", verbose=verbose)
    block_size = 1048576
    log(text=f"Decoding source file...", verbose=verbose)
    with codecs.open(source, "r", encoding=decoding) as source_file:
        log(text=f"Encoding to target file...", verbose=verbose)
        with codecs.open(target, "w", encoding=encoding) as target_file:
            while True:
                contents = source_file.read(block_size)

                if not contents:
                    break

                target_file.write(contents)

        log(text=f"File has been encoded as {encoding} at \"{target}\".", verbose=verbose)


def download(url: str, location_path: str, file_name: str = None, verbose: bool = True, ignore: bool = True) -> None:
    """
    Downloads a file from a URL and saves it to the specified location.

    Args:
        url (str): The URL of the file to be downloaded.
        location_path (str): The path to the directory where the file will be saved.
        file_name (str, optional): The name of the file to be saved. If not provided, the filename from the URL is used. Defaults to None.
        verbose (bool, optional): If True, displays log messages. Defaults to True.
        ignore (bool, optional): If True, ignores the download if the file already exists. Defaults to True.

    Returns:
        None
    """
    log(text=f"Checking if destination path \"{location_path}\" exists...", verbose=verbose)

    if not os.path.exists(location_path):
        log(text=f"Destination path does not exist, creating this path...", verbose=verbose)
        os.makedirs(location_path)

    if file_name is None:
        log(text=f"No file name given, using the url filename instead.", verbose=verbose)
        file_name = url.split("/")[-1].replace(" ", "_")

    file_path = os.path.join(location_path, file_name)

    if os.path.isfile(file_path):
        if not ignore:
            log(text=f"The fiel \"{file_path}\" already exists. Make sure this does not exist.", level=LogLevel.ERROR)
            return

        log(text=f"The file \"{file_path}\" already exists. We assume this to be the expected file and ignore the download.")
        return

    log(text=f"Downloading file from \"{url}\"...", verbose=verbose)
    req = requests.get(url, stream=True)

    if req.ok:
        with open(file_path, 'wb') as f:
            log(text=f"Saving downloaded file to \"{file_path}\"...", verbose=verbose)

            for chunk in req.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())

            log(text=f"File has been saved to \"{file_path}\".", verbose=verbose)
    else:
        log(text=f"Failed to download file from \"{url}\". Status code: {req.status_code} with message \"{req.text}\".", level=LogLevel.ERROR)


def get_or_create_directory(path: str, error: bool = False) -> str:
    """
    Retrieves an existing directory or creates a new one if it doesn't exist.

    Args:
        path (str): The path of the directory.
        error (bool, optional): If True, logs an error message if the directory doesn't exist. Defaults to False.

    Returns:
        str: The path of the existing or newly created directory.
    """
    if path is None or not isinstance(path, str):
        log(text=f"Invalid path given: \"{path}\".", level=LogLevel.ERROR)

    if not os.path.exists(path):
        if error:
            log(text=f"Directory at \"{path}\" does not exist. Please define a directory that already exists!", level=LogLevel.ERROR)

        os.makedirs(path)

    return path


def create_path(directory: str, file_name: str, create: bool = True) -> str or None:
    """
    Creates a full path by combining a directory and a file name.

    Args:
        directory (str): The directory path.
        file_name (str): The name of the file.
        create (bool, optional): If True, creates the directory if it doesn't exist. Defaults to True.

    Returns:
        str or None: The full path if both directory and file name are provided, otherwise None.
    """
    if file_name is None:
        return None

    if directory is None:
        return file_name

    get_or_create_directory(directory, error=not create)

    return os.path.join(directory, file_name)


def load_csv_with_x_and_y(path: str, class_header: str, verbose: bool = False) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads a CSV-based dataset from a file, separating the features (X) and the target variable (y).

    Args:
        path (str): The path to the CSV file.
        class_header (str): The column header representing the target variable.
        verbose (bool, optional): If True, displays log messages. Defaults to False.

    Returns:
        (pd.DataFrame, pd.DataFrame): A tuple containing the features (X) and the target variable (y) as Pandas DataFrames.
    """
    log(text=f"Loading CSV-based dataset...", verbose=verbose)

    if not os.path.isfile(path):
        log(text=f"The file located at \"{path}\" could not be found.", level=LogLevel.ERROR)

    data = pd.read_csv(path)

    X = data.drop(class_header, axis=1)
    y = data[class_header]

    return X, y




