from .aminoacids import PROTEIN_AA


class State:

    def __init__(self, pose, label):
        pass

    def get_state_sequence(self, by_chains=False) -> str:
        pass

    def _get_pose_to_pdb(self) -> dict:
        pass

    def _get_pdb_to_pose(self) -> dict:
        pass

    def _get_pdb_to_sequence(self) -> dict:
        pass

    def __repr__(self):
        pass

    def __str__(self):
        """
        Prints the label associated with the MSD state.
        """
        pass
