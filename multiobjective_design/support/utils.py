import pandas as pd


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


def _readSequencesFromFasta(fasta_file) -> dict:
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
