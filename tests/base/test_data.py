import unittest

import pandas as pd
from Bio.Seq import Seq
from pandas.testing import assert_frame_equal

from mood.base.data import AlgorithmDataSingleton

SEQUENCES_C = {
    "A": {
        Seq("ATG"): None,
        Seq("TGG"): (("A", 1, "T"), ("T", 2, "G")),
        Seq("GAT"): (("A", 1, "G"), ("T", 2, "A"), ("G", 3, "T")),
    }
}
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
        new_sequences = {
            "A": {
                Seq("CGA"): (("A", 1, "C"), ("T", 2, "G"), ("G", 3, "A")),
                Seq("TAC"): (("A", 1, "T"), ("T", 2, "A"), ("G", 3, "C")),
            }
        }
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
        self.data.add_sequence(chain="A", new_sequence=new_sequence, mut=None)

        expected_sequences = {
            "A": {
                Seq("ATG"): None,
                Seq("TGG"): (("A", 1, "T"), ("T", 2, "G")),
                Seq("GAT"): (("A", 1, "G"), ("T", 2, "A"), ("G", 3, "T")),
                Seq("CGA"): None,
            }
        }
        self.assertEqual(self.data.sequences, expected_sequences)

    def test_add_sequence_same(self):
        """Test adding a single sequence."""
        new_sequence = Seq("ATG")
        self.data.add_sequence(chain="A", new_sequence=new_sequence)

        expected_sequences = {
            "A": {
                Seq("ATG"): None,
                Seq("TGG"): (("A", 1, "T"), ("T", 2, "G")),
                Seq("GAT"): (("A", 1, "G"), ("T", 2, "A"), ("G", 3, "T")),
                Seq("CGA"): None,
            }
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
        instance1 = AlgorithmDataSingleton(sequences={"A": Seq("ATG")})
        self.assertEqual(instance1.sequences, SEQUENCES_C)

    def test_empty_sequences(self):
        """Test behaviour with empty sequences."""
        self.data.replace_data(sequences={})
        self.assertEqual(self.data.sequences, {})


if __name__ == "__main__":
    unittest.main()
