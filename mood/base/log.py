import logging
import os
import threading


class SingletonType(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonType, cls).__call__(
                        *args, **kwargs
                    )
        return cls._instances[cls]


class Logger(metaclass=SingletonType):
    _log_lock = threading.Lock()

    def __init__(self, debug=False, log_dir=None, filename="mood.log"):
        with self._log_lock:
            # Ensure only one initialization
            if not hasattr(self, "_log_initialized"):
                # Create log directory if it doesn't exist
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                    full_path = os.path.join(log_dir, filename)
                else:
                    full_path = filename

                # Create logger
                self.log = logging.getLogger("global_logger")
                (
                    self.log.setLevel(logging.DEBUG)
                    if debug
                    else self.log.setLevel(logging.INFO)
                )

                # Clear any existing handlers to prevent duplicates
                if self.log.handlers:
                    for handler in self.log.handlers[:]:
                        self.log.removeHandler(handler)

                # Create file handler with mode='a' for appending
                file_handler = logging.FileHandler(full_path, mode="a")
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%d-%m-%y %H:%M:%S",
                )
                file_handler.setFormatter(file_formatter)

                # Add handler
                self.log.addHandler(file_handler)

                # Prevent re-initialization
                self._log_initialized = True

    def get_log(self):
        return self.log
