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


def log(text: str, level: LogLevel = LogLevel.DEFAULT, verbose=True) -> None:
    """
    Print log messages with a specified log level.

    Args:
        text (str): The log message text to be printed.
        level (LogLevel, optional): The log level of the message. Defaults to LogLevel.DEFAULT.
        verbose (bool, optional): Controls whether the log message should be printed.
            If set to False, the function will return without printing anything.
            Defaults to True.

    Returns:
        None

    Log Levels:
        The log levels are represented using the LogLevel enum.

        - LogLevel.DEFAULT: The default log level.
        - LogLevel.INFO: Informational log level.
        - LogLevel.WARNING: Warning log level.
        - LogLevel.ERROR: Error log level.
        - LogLevel.OK: Success log level.

    Note:
        - If verbose is set to False, the log message will not be printed, regardless of the log level, except for errors, they will always be forced.
        - If the log level is set to LogLevel.ERROR, an Exception will be raised instead of printing the log message.
    """

    if level == LogLevel.ERROR:
        raise Exception(f"{level.value} {text}{DEFAULT}")

    if not verbose:
        return

    print(f"{level.value} {text}{DEFAULT}")
