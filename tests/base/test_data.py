import unittest

import pandas as pd
from Bio.Seq import Seq
from pandas.testing import assert_frame_equal

from mood.base.data import AlgorithmDataSingleton

SEQUENCES_C = {0: Seq("ATG"), 1: Seq("TGA"), 2: Seq("GAT")}
DATA_FRAME = pd.DataFrame(
    {
        "seq_index": [0, 1, 2],
        "Sequence": [Seq("ATG"), Seq("TGA"), Seq("GAT")],
        "iteration": [1, 1, 2],
        "Metric1": [0.1, 0.2, 0.3],
        "Metric2": [1.1, 1.2, 1.3],
    }
)


class AlgorithmDataSingletonTests(unittest.TestCase):
    def setUp(self):
        self.data = AlgorithmDataSingleton(
            sequences=SEQUENCES_C.copy(), data_frame=DATA_FRAME.copy()
        )

    def test_replace_data(self):
        """Test replacing data in the singleton."""
        new_sequences = {4: Seq("CGA"), 5: Seq("TAC")}
        new_data_frame = pd.DataFrame(
            {
                "seq_index": [4, 5],
                "Sequence": [Seq("CGA"), Seq("TAC")],
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
        self.data.replace_data(sequences, DATA_FRAME)
        self.data.add_sequences_from_df()
        from icecream import ic

        ic(self.data.sequences)
        ic(SEQUENCES_C)

        self.assertEqual(self.data.sequences, SEQUENCES_C)

    def test_add_sequence(self):
        """Test adding a single sequence."""
        new_sequence = Seq("CGA")
        self.data.add_sequence(new_sequence)

        expected_sequences = {
            0: Seq("ATG"),
            1: Seq("TGA"),
            2: Seq("GAT"),
            3: Seq("CGA"),
        }
        self.assertEqual(self.data.sequences, expected_sequences)

    def test_get_data(self):
        """Test retrieving data from the singleton."""

        self.data.replace_data(sequences=SEQUENCES_C, data_frame=DATA_FRAME)

        sequences, data_frame = self.data.get_data()

        self.assertEqual(sequences, SEQUENCES_C)
        assert_frame_equal(data_frame, DATA_FRAME)

    def test_singleton_instance_created_once(self):
        """Test that the singleton instance is created only once."""
        instance1 = AlgorithmDataSingleton(sequences=SEQUENCES_C, data_frame=None)
        instance2 = AlgorithmDataSingleton(sequences=SEQUENCES_C, data_frame=None)
        self.assertIs(instance1, instance2)

    def test_multiple_instances_return_same_singleton(self):
        """Test that multiple instances return the same singleton."""
        instance1 = AlgorithmDataSingleton(sequences={"0": "seq1"}, data_frame=None)
        self.assertEqual(instance1.sequences, SEQUENCES_C)

    def test_empty_sequences(self):
        """Test behaviour with empty sequences."""
        self.data.replace_data(sequences={}, data_frame=DATA_FRAME)
        self.assertEqual(self.data.sequences, {})

    def test_empty_data_frame(self):
        """Test behaviour with an empty data frame."""
        empty_df = pd.DataFrame()
        self.data.replace_data(sequences=SEQUENCES_C, data_frame=empty_df)
        assert_frame_equal(self.data.data_frame, empty_df)

    def test_clear_dataFrame(self):
        """Test clearing the dataFrame."""
        self.data.clear_dataFrame()
        assert_frame_equal(
            self.data.data_frame,
            pd.DataFrame(columns=["seq_index", "Sequence", "iteration"]),
        )

    def test_add_info_df(self):
        """Test adding information to the data frame."""
        self.data.add_info_df(
            {
                "seq_index": 3,
                "Sequence": Seq("KAK"),
                "iteration": 3,
            }
        )
        import numpy as np

        expected_data_frame = pd.DataFrame(
            {
                "seq_index": [0, 1, 2, 3],
                "Sequence": [Seq("ATG"), Seq("TGA"), Seq("GAT"), Seq("KAK")],
                "iteration": [1, 1, 2, 3],
                "Metric1": [0.1, 0.2, 0.3, np.nan],
                "Metric2": [1.1, 1.2, 1.3, np.nan],
            }
        )
        assert_frame_equal(self.data.data_frame, expected_data_frame)


if __name__ == "__main__":
    unittest.main()
