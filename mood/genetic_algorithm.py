import logging
import os
import random
import shutil
import subprocess
from math import floor

import numpy as np
import pandas as pd

from .base import Sequence, Silent
from .utils import analysis_geneticAlgorithm


class geneticAlgorithm:

    def __init__(
        self,
        job_folder="GA",
        population_size=100,
        mutation_rate=0.005,
        elites_fraction=0.00,
        max_mutations=None,
        seed=None,
        debug=False,
    ):
        pass

    def deactivateParents(self):
        pass

    def runGeneticAlgorithm(
        self,
        mutated_seq_pct=0.5,
    ):
        pass

    def createPopulationByRecombination(self, max_attempts=1000, verbose=True) -> bool:
        pass

    def mutatePopulation(self, verbose=True):
        pass

    def mutateSequence(self, sequence, return_aa_info=False) -> Sequence:
        pass

    def recombineSequences(self, sequence1, sequence2, skip_assert=False) -> Sequence:
        pass

    def revertSequence(self, sequence) -> Sequence:
        pass

    def mutateResidue(self, pose, mutant_position, mutant_aa):
        pass

    def _getMutatedPositions(self, native_sequence, target_sequence) -> dict:
        pass

    def _create_first_population(
        self, sequence_init, chain, n_missing, mutated_seq_pct
    ):
        pass

    def get_seed(self) -> int:
        pass

    def set_seed(self, seed):
        pass


def round_half_up(n, decimals=0):
    pass
