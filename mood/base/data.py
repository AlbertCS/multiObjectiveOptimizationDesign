import logging

import pandas as pd


class AlgorithmDataSingleton:
    _instance = None

    """
    sequences: {"chain: {"index": "sequence"}}
    TODO look if we want to track the ancestry of the sequences, fathers, grandfathers, etc.
    For now, the sequences dictionary will be persistent and the dataFrame will be reset with each iteration.
    """

    def __new__(cls, sequences={}, chains=[]):
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

    @chains.setter
    def chains(self, chains):
        self._chains = chains
        self.sequences = {chain: {} for chain in self.chains}

    def nsequences(self, chain):
        return len(self.sequences.get(chain))

    def clear_data(self):
        """
        Clear the existing sequences and data frame in the AlgorithmDataSingleton instance.
        Reset sequences to an empty dictionary and create an empty DataFrame with columns ["seq_index", "Sequence", "iteration"].
        """
        self._sequences = {}

    def add_sequences(self, chain, new_sequences):
        """Returns True if the sequence was added successfully, False otherwise."""
        for sequence in new_sequences:
            if sequence.sequence in self.sequences.get(chain).keys():
                continue
            else:
                self.sequences[chain][sequence.sequence] = sequence

    def sequence_exists(self, chain, sequence):
        # Returns True if the sequence exists in the chain, False otherwise.
        return sequence in self.sequences.get(chain).keys()
