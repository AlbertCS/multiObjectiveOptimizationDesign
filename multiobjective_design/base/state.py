from .aminoacids import PROTEIN_AA


class State:

    def __init__(self, pose, label):
        pass

    def getStateSequence(self, by_chains=False) -> str:
        pass

    def _getPose2PDB(self) -> dict:
        pass

    def _getPDB2Pose(self) -> dict:
        pass

    def _getPDB2Sequence(self) -> dict:
        pass

    def __repr__(self):
        pass

    def __str__(self):
        """
        Prints the label associated with the MSD state.
        """
        pass
