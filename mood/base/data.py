import logging

import pandas as pd


class AlgorithmDataSingleton:
    _instance = None

    """
    sequences: {"index": "sequence"}
    data_frame: pd.DataFrame(columns=["seq_index", "Sequence", "iteration", "Metric1", "Metric2", ...])
    TODO look if we want to track the ancestry of the sequences, fathers, grandfathers, etc.
    
    For now, the sequences dictionary will be persistent and the dataFrame will be reset with each iteration.
    """

    def __new__(cls, sequences=None, data_frame=None):
        if cls._instance is None:
            cls._instance = super(AlgorithmDataSingleton, cls).__new__(cls)
            # Initialize the instance attributes
            cls._instance.sequences = sequences
            cls._instance.data_frame = data_frame
        return cls._instance

    @property
    def sequences(self):
        return self._sequences

    @property
    def data_frame(self):
        return self._data_frame

    @sequences.setter
    def sequences(self, sequences):
        self._sequences = sequences

    @data_frame.setter
    def data_frame(self, data_frame):
        self._data_frame = data_frame

    def clear_data(self):
        """
        Clear the existing sequences and data frame in the AlgorithmDataSingleton instance.
        Reset sequences to an empty dictionary and create an empty DataFrame with columns ["seq_index", "Sequence", "iteration"].
        """
        self._sequences = {}
        self.data_frame = pd.DataFrame(columns=["seq_index", "Sequence", "iteration"])

    def clear_dataFrame(self):
        """
        Clear the existing data frame in the AlgorithmDataSingleton instance.
        Create an empty DataFrame with columns ["seq_index", "Sequence", "iteration"].
        """
        self.data_frame = pd.DataFrame(columns=["seq_index", "Sequence", "iteration"])

    def replace_data(self, sequences, data_frame):
        """
        Replace the existing sequences and data frame in the AlgorithmDataSingleton instance with new ones.

        Parameters:
        sequences: {"index": "sequence"} - New sequences to replace the existing ones.
        data_frame: pd.DataFrame(columns=["seq_index", "Sequence", "iteration", "Metric1", "Metric2", ...]) - New data frame to replace the existing one.
        """
        self.sequences = sequences
        self.data_frame = data_frame

    def add_sequences_from_df(self):
        """
        Add sequences from the data frame to the existing sequences in the AlgorithmDataSingleton instance.

        This method extracts the "Sequence" column from the data frame and converts it into a dictionary format to update the sequences attribute.

        This method does not return any value.
        """
        self.sequences = self.data_frame["Sequence"].to_dict()

    def add_sequence(self, new_sequence) -> bool:
        """Returns True if the sequence was added successfully, False otherwise."""
        if new_sequence in self.sequences.values():
            logging.warning(f"Sequence {new_sequence} already in the data")
            return False
        else:
            self.sequences[len(self.sequences)] = new_sequence
            return True

    def get_data(self):
        return self.sequences, self.data_frame
