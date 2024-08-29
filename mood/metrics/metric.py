from abc import ABC, abstractmethod


class Metric:
    def __init__(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def setup_iterations_inputs(self):
        pass
