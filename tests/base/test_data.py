import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from mood.base.data import AlgorithmDataSingleton


class AlgorithmDataSingletonTests(unittest.TestCase):
    def setUp(self):
        sequences = {0: "ATG", 1: "TGA", 2: "GAT"}
        seq_index = [0, 1, 2]
        sequence = ["ATG", "TGA", "GAT"]
        iteration = [1, 1, 2]
        metric1 = [0.1, 0.2, 0.3]
        metric2 = [1.1, 1.2, 1.3]

        data_frame = pd.DataFrame(
            {
                "seq_index": seq_index,
                "Sequence": sequence,
                "iteration": iteration,
                "Metric1": metric1,
                "Metric2": metric2,
            }
        )

        self.data = AlgorithmDataSingleton(sequences=sequences, data_frame=data_frame)

    def test_replace_data(self):
        """Test replacing data in the singleton."""
        new_sequences = {4: "CGA", 5: "TAC"}
        new_data_frame = pd.DataFrame(
            {
                "seq_index": [4, 5],
                "Sequence": ["CGA", "TAC"],
                "iteration": [3, 3],
                "Metric1": [0.4, 0.5],
                "Metric2": [1.4, 1.5],
            }
        )

        self.data.replace_data(new_sequences, new_data_frame)

        self.assertEqual(self.data.sequences, new_sequences)
        assert_frame_equal(self.data.data_frame, new_data_frame)

    def test_add_sequences_from_df(self):
        """Test adding sequences from the data frame."""
        sequences = {}
        self.data.replace_data(sequences=sequences, data_frame=self.data.data_frame)
        self.data.add_sequences_from_df()

        expected_sequences = {0: "ATG", 1: "TGA", 2: "GAT"}
        self.assertEqual(self.data.sequences, expected_sequences)

    def test_add_sequence(self):
        """Test adding a single sequence."""
        new_sequence = "CGA"
        self.data.add_sequence(new_sequence)

        expected_sequences = {0: "ATG", 1: "TGA", 2: "GAT", 3: "CGA"}
        self.assertEqual(self.data.sequences, expected_sequences)

    def test_get_data(self):
        """Test retrieving data from the singleton."""

        expected_sequences = {0: "ATG", 1: "TGA", 2: "GAT"}
        expected_data_frame = pd.DataFrame(
            {
                "seq_index": [0, 1, 2],
                "Sequence": ["ATG", "TGA", "GAT"],
                "iteration": [1, 1, 2],
                "Metric1": [0.1, 0.2, 0.3],
                "Metric2": [1.1, 1.2, 1.3],
            }
        )
        self.data.replace_data(
            sequences=expected_sequences, data_frame=expected_data_frame
        )

        sequences, data_frame = self.data.get_data()

        self.assertEqual(sequences, expected_sequences)
        assert_frame_equal(data_frame, expected_data_frame)

    def test_singleton_instance_created_once(self):
        """Test that the singleton instance is created only once."""
        instance1 = AlgorithmDataSingleton(
            sequences={0: "ATG", 1: "TGA", 2: "GAT"}, data_frame=None
        )
        instance2 = AlgorithmDataSingleton(
            sequences={0: "ATG", 1: "TGA", 2: "GAT"}, data_frame=None
        )
        self.assertIs(instance1, instance2)

    def test_multiple_instances_return_same_singleton(self):
        """Test that multiple instances return the same singleton."""
        instance1 = AlgorithmDataSingleton(sequences={"0": "seq1"}, data_frame=None)
        self.assertEqual(instance1.sequences, {0: "ATG", 1: "TGA", 2: "GAT"})

    def test_empty_sequences(self):
        """Test behaviour with empty sequences."""
        self.data.replace_data(sequences={}, data_frame=self.data.data_frame)
        self.assertEqual(self.data.sequences, {})

    def test_empty_data_frame(self):
        """Test behaviour with an empty data frame."""
        empty_df = pd.DataFrame()
        self.data.replace_data(sequences=self.data.sequences, data_frame=empty_df)
        assert_frame_equal(self.data.data_frame, empty_df)

    def test_clear_dataFrame(self):
        """Test clearing the dataFrame."""
        self.data.clear_dataFrame()
        assert_frame_equal(
            self.data.data_frame,
            pd.DataFrame(columns=["seq_index", "Sequence", "iteration"]),
        )


if __name__ == "__main__":
    unittest.main()
