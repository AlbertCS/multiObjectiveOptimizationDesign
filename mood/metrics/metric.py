from abc import ABC, abstractmethod


class Metric:
    def __init__(self, iteration_folder, data=None):
        self.data = data

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def setup_iterations_inputs(self):
        pass
