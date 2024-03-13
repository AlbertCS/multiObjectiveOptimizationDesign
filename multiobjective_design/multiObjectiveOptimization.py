from pyrosetta import Pose, Vector1, init, pose_from_file

init()


class moo:

    def __init__(self):
        """
        Initilises the MSD class
        """
        self.states = {}
        self.positive_states = []
        self.negative_states = []
        self.sequences = []  # list of sequences to not repeat them
        self.n_states = 0
        self.n_sequences = {}
        self.active_chains = []
        self.mutatable_positions = {}
        self.allowed_aminoacids = {}
        self.params_files = []
        self.native_sequence = None
        self.metrics = {}
        self.objectives = {}

    def addStateFromPose(
        self,
        pose,
        label,
        state_index=None,
        overwrite=False,
        negative_state=False,
        chains=None,
    ):
        pass

    def addStateFromPDB(
        self,
        pdb_file,
        label=None,
        params_files=None,
        state_index=None,
        negative_state=False,
        overwrite=False,
        chains=None,
    ):

        pass

    def addSequences(
        self,
        sequences,
        chain,
        sequence_index=None,
        parent=False,
        recombined=False,
        child=False,
        active=True,
        mutated=False,
        reverted=False,
        verbose=True,
        replace=False,
    ):
        pass

    def getNewSequenceIndex(self, chain) -> int:
        pass

    def addSequencesFromMutations(
        self, mutants, chain, reference_state=1, child=True, parent=True, active=True
    ) -> list[Sequence]:
        pass

    def addSequencesFromFasta(
        self, fasta_file, chain, child=True, parent=True, active=True
    ):
        pass

    def addStateSequence(self, state, child=False, parent=True, active=True):

        pass

    def setNativeSequence(self, sequence=None, state_index=None):
        pass

    def removeSequence(self, sequence_index, chain):
        pass

    def getSequenceByLabel(self, label, chain) -> Sequence:
        pass

    def setMutatablePositions(
        self, residue_positions, chain, not_native_aa=False
    ) -> dict:
        pass

    def getSequencesWithEnergy(self, return_strings=True) -> dict:
        pass

    def getSequencesPool(
        self, chain, active=None, recombined=None, mutated=None, parent=None, child=None
    ) -> dict:
        pass

    def _checkIfSequenceExists(self, sequence, chain=None) -> int:
        pass

    def non_dominated_sorting(self, population):
        pass
