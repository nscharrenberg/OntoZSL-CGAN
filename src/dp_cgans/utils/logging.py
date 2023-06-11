from enum import Enum
from colorama import Fore

ERROR = Fore.RED
INFO = Fore.CYAN
WARNING = Fore.YELLOW
OK = Fore.GREEN
DEFAULT = Fore.RESET


class LogLevel(Enum):
    ERROR = ERROR
    INFO = INFO
    WARNING = WARNING
    OK = OK
    DEFAULT = DEFAULT


def log(text: str, level: LogLevel = LogLevel.DEFAULT, verbose: bool = True):
    """
    When allowed, print a log with the given message and log level.

    :param text: The message to log
    :param level: The level of the log (DEBUG = 0, OK = 10, INFO = 20, WARN = 30, ERROR = 40, CRITICAL = 50)
    :param verbose: Whether to print to console or not.
    """
    if not verbose:
        return

    print(f"{level.value} {text}{DEFAULT}")


def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
