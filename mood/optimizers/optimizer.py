from abc import abstractmethod
from enum import Enum

import pandas as pd

from mood.base.data import AlgorithmDataSingleton


class OptimizersType(str, Enum):
    """
    The different optimizers available in the package.
    """

    GA = "genetic_algorithm"
    """
    Genetic Algorithm optimizer.
    """

    def __str__(self):
        return self.value


class Optimizer:

    def __init__(
        self,
        population_size=100,
        seed=12345,
        debug=False,
        data=None,
        optimizerType: OptimizersType = OptimizersType.GA,
        mutable_positions=None,
        mutable_aa=None,
    ):
        self.population_size = population_size
        self.seed = seed
        self.debug = debug
        self.data = (
            data
            if data is not None
            else AlgorithmDataSingleton(
                sequences={},
                chains=[],
            )
        )
        if optimizerType not in (OptimizersType):
            raise Exception(  # pylint: disable=broad-exception-raised
                f"Invalid optimizer type {optimizerType}. Allowed types: {OptimizersType}"
            )
        self.TYPE: OptimizersType = optimizerType
        self.native = None

    @property
    def native(self):
        return self._native

    @native.setter
    def native(self, native):
        self._native = native

    @abstractmethod
    def init_population(self):
        pass

    @abstractmethod
    def eval_population(self, population):
        pass

    @abstractmethod
    def generate_child_population(self, parent_sequences):
        pass
