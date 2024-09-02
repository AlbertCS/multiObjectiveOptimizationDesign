import unittest

import pandas as pd
from Bio.Seq import Seq
from pandas.testing import assert_frame_equal

from mood.base.data import AlgorithmDataSingleton

SEQUENCES_C = {"A": {0: Seq("ATG"), 1: Seq("TGA"), 2: Seq("GAT")}}
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
        self.data = AlgorithmDataSingleton(sequences=SEQUENCES_C.copy())

    def test_replace_data(self):
        """Test replacing data in the singleton."""
        new_sequences = {"A": {4: Seq("CGA"), 5: Seq("TAC")}}
        self.data.replace_data(new_sequences)

        self.assertEqual(self.data.sequences, new_sequences)

    def test_add_sequences_from_df(self):
        """Test adding sequences from the data frame."""
        sequences = {}
        self.data.replace_data(sequences)
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

        self.data.replace_data(sequences=SEQUENCES_C)

        sequences = self.data.get_data()

        self.assertEqual(sequences, SEQUENCES_C)

    def test_singleton_instance_created_once(self):
        """Test that the singleton instance is created only once."""
        instance1 = AlgorithmDataSingleton(sequences=SEQUENCES_C)
        instance2 = AlgorithmDataSingleton(sequences=SEQUENCES_C)
        self.assertIs(instance1, instance2)

    def test_multiple_instances_return_same_singleton(self):
        """Test that multiple instances return the same singleton."""
        instance1 = AlgorithmDataSingleton(sequences={"0": "seq1"})
        self.assertEqual(instance1.sequences, SEQUENCES_C)

    def test_empty_sequences(self):
        """Test behaviour with empty sequences."""
        self.data.replace_data(sequences={})
        self.assertEqual(self.data.sequences, {})


if __name__ == "__main__":
    unittest.main()
