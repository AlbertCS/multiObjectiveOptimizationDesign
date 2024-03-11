import os
import random
import shutil
import sys
from typing import List


def setUpMSD(
    job_folder,
    msd_script,
    objectives="total_score",
    population_size=100,
    replicas=10,
    iterations=200,
    cpus=48,
    relax_cycles=5,
    neighbour_distance=12.0,
    parallelization="slurm",
    max_mutations=None,
    bias_type="max_energy",
    n_attempts=1,
    additional_folders=None,
    msa_restricted=False,
    seed=None,
    not_native_aa=False,
) -> List[str]:
    pass
