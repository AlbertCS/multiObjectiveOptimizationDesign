from abc import abstractmethod
from enum import Enum


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
        mutation_rate=0.005,
        seed=12345,
        debug=False,
        data=None,
        optimizerType: OptimizersType = OptimizersType.GA,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.debug = debug
        self.data = data
        if optimizerType not in (OptimizersType):
            raise Exception(  # pylint: disable=broad-exception-raised
                f"Invalid optimizer type {optimizerType}. Allowed types: {OptimizersType}"
            )
        self.TYPE: OptimizersType = optimizerType

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
