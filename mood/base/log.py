import logging
import threading


class SingletonType(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking pattern
        if cls not in cls._instances:
            with cls._lock:
                # Another thread might have created the instance
                # while the first thread was waiting for the lock
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonType, cls).__call__(
                        *args, **kwargs
                    )
        return cls._instances[cls]


class Logger(metaclass=SingletonType):
    _log_lock = threading.Lock()

    def __init__(self, debug=False, timestamp=""):
        with self._log_lock:
            # Use a process ID to ensure uniqueness across threads/processes
            import os

            pid = os.getpid()

            # Create logger only if it doesn't exist
            if not hasattr(self, "_log_initialized"):
                self.log = logging.getLogger(f"mood_logger_{pid}")
                (
                    self.log.setLevel(logging.DEBUG)
                    if debug
                    else self.log.setLevel(logging.INFO)
                )

                # FileHandler for file output
                log_filename = f"mood_{timestamp}_pid{pid}.log"
                file_handler = logging.FileHandler(log_filename, mode="w")
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%d-%m-%y %H:%M:%S",
                )
                file_handler.setFormatter(file_formatter)

                # Remove any existing handlers to prevent duplicate logging
                self.log.handlers.clear()
                self.log.addHandler(file_handler)

                # Mark as initialized
                self._log_initialized = True

    def get_log(self):
        return self.log
