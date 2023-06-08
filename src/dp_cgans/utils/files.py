import codecs
import os

import pandas as pd
import requests

from dp_cgans.utils.logging import log


def recode(source: str, target: str, decoding: str = "ANSI", encoding: str = "utf-8", verbose: bool = False) -> None:
    """
    Converts the encoding of a file from the defined {decoding} to {encoding} and saving it from {source} to {target}.
    Args:
        source: The file to be recoded
        target: The file path for the newly encoded file to be saved to.
        decoding: The type of encoding that was used for the source file.
        encoding: The type of encoding you want to use to recode the file as.
        verbose: Whether to print intermediate information messages.
    """
    log(text=f'Preparing file to be  from {decoding} to {encoding}...', verbose=verbose)
    block_size = 1048576
    log(text=f'Decoding source file....', verbose=verbose)
    with codecs.open(source, "r", encoding=decoding) as sourceFile:

        log(text=f'Encoding to target file...', verbose=verbose)
        with codecs.open(target, "w", encoding=encoding) as targetFile:
            while True:
                contents = sourceFile.read(block_size)
                if not contents:
                    break
                targetFile.write(contents)

    log(text=f'File has been encoded as {encoding} at \"{target}\"', verbose=verbose)


def download(url: str, location_path: str, file_name: str = None, verbose: bool = True, ignore: bool = True) -> None:
    """
    Retrieve an external file from the internet to your computer.
    Args:
        url: The file url to be retrieved
        location_path: the directory to save the file to
        file_name: the filename the file should have including the extension
        verbose: Whether to print intermediate information messages.

    """
    log(text=f'Checking if destination path "{location_path}" exists...', verbose=verbose)
    if not os.path.exists(location_path):
        log(text=f'Destination path does not exist, creating this path...', verbose=verbose)
        os.makedirs(location_path)

    if file_name is None:
        log(text=f'No file name given, using the url filename instead!', verbose=verbose)
        file_name = url.split('/')[-1].replace(" ", "_")

    file_path = os.path.join(location_path, file_name)

    if os.path.isfile(file_path):
        if not ignore:
            raise IOError(f"The file {file_path} already exists. Make sure this does not exist!")
        log(text=f"The file \"{file_path}\" already exists. We assume this to be the expected file and ignore the download.")
        return

    log(text=f'🔄️  Downloading file from {url}...', verbose=verbose)
    req = requests.get(url, stream=True)

    if req.ok:
        with open(file_path, 'wb') as f:
            log(text=f'💾️  Saving downloaded file to {file_path}...', verbose=verbose)
            for chunk in req.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
            log(text=f'✅️Success! File has been saved to {file_path}.', verbose=True)
    else:
        raise Exception(f"Failed to download file from {url}. Status code: {req.status_code} with message: {req.text}")


def create_path(directory: str, file_name: str, create: bool = True):
    if file_name is None:
        return None

    get_or_create_directory(directory, error=not create)

    return os.path.join(directory, file_name)


def get_or_create_directory(path: str, error: bool = False):
    """
    Checks if the path exists or not. If it does not exist it'll either throw an exception or create the directory.
    Args:
        path: The directory to check
        error: If true an exception will be thrown when the directory does not exist, otherwise the directory will be created.

    Returns: the directory path

    """
    if path is None or not isinstance(path, str):
        raise Exception(f"Invalid path given: \"{path}\".")

    if not os.path.exists(path):
        if error:
            raise Exception(f"Directory at \"{path}\" does not exist. Please define a directory that already exists!")

        os.makedirs(path)

    return path


def load_csv(path: str, class_header: str, verbose: bool = False):
    log(f"Reading \"Seen\" dataset...", verbose=verbose)

    if not os.path.isfile(path):
        raise IOError(f"The file located at \"{path}\" could not be found")

    seen_data = pd.read_csv(path)

    X = seen_data.drop(class_header, axis=1)
    y = seen_data[class_header]

    return X, y
