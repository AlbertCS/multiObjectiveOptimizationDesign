import logging
import os
import random
import shutil
import subprocess
from math import floor

import numpy as np
import pandas as pd
from base import Sequence, Silent
from pyrosetta import *
from support import analysis_geneticAlgorithm

from . import rosettaScripts as rs


class geneticAlgorithm:

    def __init__(
        self,
        msd,
        objectives="total_score",
        job_folder="GA",
        population_size=100,
        mutation_rate=0.005,
        xml_protocol=None,
        iterations=100,
        bias_type="max_energy",
        elites_fraction=0.00,
        max_mutations=None,
        KT_min=0.5,
        KT_update=10,
        seed=None,
        debug=False,
        distances=None,
        rosetta_seed=None,
    ):
        pass

    def readIterationScores(self, iterations) -> pd.DataFrame:
        pass

    def checkPreviousIterations(self):
        pass

    def checkIfIterationIsFinished(self, iteration) -> bool:
        pass

    def checkIfIterationIsMutated(self) -> bool:
        pass

    def addIterationDataToMSD(
        self, iteration, active=True, parent=False, child=False, verbose=True
    ):
        pass

    def deactivateParents(self):
        pass

    def runGeneticAlgorithm(
        self,
        parallelization="slurm",
        cpus=32,
        executable="rosetta_scripts.mpi.linuxgccrelease",
        mutated_seq_pct=0.5,
    ):
        pass

    def setIterationFolders(self, iteration):
        pass

    def setIterationData(self, iteration, overwrite=False):
        pass

    def setUpIterationInputs(
        self,
        iteration,
        relax_options=True,
        nstruct=1,
        executable="rosetta_scripts.mpi.linuxgccrelease",
        parallelization=None,
        cpus=1,
    ) -> list[str]:
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

    def createPopulationSilentFile(self, state_index, silent_file) -> Silent:
        pass

    def mutateResidue(self, pose, mutant_position, mutant_aa):
        pass

    def setXMLProtocol(self, xml_object):
        pass

    def selectParentsByNonDominatedRank(
        self,
        iteration,
        objectives,
        percentage=50.0,
        verbose=True,
        bias_type="max_energy",
        KT_min=0.5,
        KT_update=10,
        max_energy=10,
        max_energy_state=1,
        distances=None,
    ):
        pass

    def assignCrowdingDistance(self, data, objectives) -> pd.DataFrame:
        pass

    def assingBoltzmannBias(self, data, objective, KT=1.2) -> pd.DataFrame:
        pass

    def getObjectivesData(self, data, objectives) -> pd.DataFrame:
        pass

    def assignNonDominatedRanks(
        self, data, objectives, distances, to_keep, filter_incr=0.5
    ) -> pd.DataFrame:
        pass

    def selectParentsByFitness(self, iteration, percentage=50.0, verbose=True):
        pass

    def addStateEnergyToSequences(self, state, score_file) -> dict:
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
