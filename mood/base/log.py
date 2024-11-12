# Python code for a logger with a singleton implementation

import logging


class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=SingletonType):
    def __init__(self, debug=False):
        self.log = logging.getLogger("mood_logger")
        self.log.setLevel(logging.DEBUG) if debug else self.log.setLevel(logging.INFO)

        # StreamHandler for console output
        # stream_handler = logging.StreamHandler()
        # stream_formatter = logging.Formatter(
        #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # )
        # stream_handler.setFormatter(stream_formatter)
        # self.log.addHandler(stream_handler)

        # FileHandler for file output
        # file_handler = logging.FileHandler("mood.log", mode="w")
        file_handler = logging.FileHandler("mood.log", mode="a")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d-%m-%y %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.log.addHandler(file_handler)

    def get_log(self):
        return self.log
