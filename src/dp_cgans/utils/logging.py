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
