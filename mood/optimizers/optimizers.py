from abc import ABC, abstractmethod


class Optimizer:

    @abstractmethod
    def init_population(self):
        pass

    @abstractmethod
    def get_sequences(self) -> list:
        pass

    @abstractmethod
    def eval_population(self, population):
        pass

    @abstractmethod
    def crossOver_mutate(self, population):
        pass
