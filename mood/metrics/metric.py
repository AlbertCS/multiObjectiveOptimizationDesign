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

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
