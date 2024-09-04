import logging

import pandas as pd


class AlgorithmDataSingleton:
    _instance = None

    """
    sequences: {"chain: {"index": "sequence"}}
    TODO look if we want to track the ancestry of the sequences, fathers, grandfathers, etc.
    For now, the sequences dictionary will be persistent and the dataFrame will be reset with each iteration.
    """

    def __new__(cls, sequences=None, chains=None):
        if cls._instance is None:
            cls._instance = super(AlgorithmDataSingleton, cls).__new__(cls)
            # Initialize the instance attributes
            cls._instance.sequences = sequences
            cls._instance.chains = chains
        return cls._instance

    @property
    def sequences(self):
        return self._sequences

    @property
    def chains(self):
        return self._sequences.keys()

    @sequences.setter
    def sequences(self, sequences):
        self._sequences = sequences

    def chains(self, chains):
        self._chains = chains
        self.sequences = {chain: {} for chain in self.chains}

    def clear_data(self):
        """
        Clear the existing sequences and data frame in the AlgorithmDataSingleton instance.
        Reset sequences to an empty dictionary and create an empty DataFrame with columns ["seq_index", "Sequence", "iteration"].
        """
        self._sequences = {}

    def clear_dataFrame(self):
        """
        Clear the existing data frame in the AlgorithmDataSingleton instance.
        Create an empty DataFrame with columns ["seq_index", "Sequence", "iteration"].
        """

    def replace_data(self, sequences):
        """
        Replace the existing sequences and data frame in the AlgorithmDataSingleton instance with new ones.

        Parameters:
        sequences: {"chain": {Seq: Mut}} - New sequences to replace the existing ones.
        """
        self.sequences = sequences

    def add_sequence(self, chain, new_sequence, mut=None) -> bool:
        """Returns True if the sequence was added successfully, False otherwise."""
        if new_sequence in self.sequences.get(chain).keys():
            return False
        else:
            self.sequences[chain][new_sequence] = mut
            return True

    def get_data(self):
        return self.sequences
