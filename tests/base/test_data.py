import unittest
from copy import deepcopy

import pandas as pd
from Bio.Seq import Seq
from icecream import ic
from pandas.testing import assert_frame_equal

from mood.base.data import AlgorithmDataSingleton
from mood.base.sequence import Sequence

SEQ1 = Sequence(
    sequence="ATG",
    chain="A",
    index=1,
    active=True,
    parent=None,
    child=None,
    native=None,
)
SEQ2 = Sequence(
    sequence="TGG",
    chain="A",
    index=2,
    active=True,
    parent=None,
    child=None,
    native=SEQ1,
)
SEQ3 = Sequence(
    sequence="GAT",
    chain="A",
    index=3,
    active=True,
    parent=None,
    child=None,
    native=SEQ1,
)
SEQUENCES_C = {
    "A": {
        SEQ1.sequence: SEQ1,
        SEQ2.sequence: SEQ2,
        SEQ3.sequence: SEQ3,
    }
}
DATA_FRAME = pd.DataFrame(
    {
        "seq_index": [0, 1, 2],
        "Sequence": ["ATG", "TGA", "GAT"],
        "iteration": [1, 1, 2],
        "Metric1": [0.1, 0.2, 0.3],
        "Metric2": [1.1, 1.2, 1.3],
    }
)


class AlgorithmDataSingletonTests(unittest.TestCase):

    def setUp(self):
        self.data = AlgorithmDataSingleton(
            sequences=deepcopy(SEQUENCES_C), chains=["A"]
        )
        self.data.sequences = deepcopy(SEQUENCES_C)

    def test_replace_data(self):
        """Test replacing data in the singleton."""
        new_sequences = {
            "A": {
                Sequence(
                    "CGA",
                    chain="A",
                    index=3,
                    active=True,
                    parent=None,
                    child=None,
                    native=SEQ1,
                ),
                Sequence(
                    "TAC",
                    chain="A",
                    index=4,
                    active=True,
                    parent=None,
                    child=None,
                    native=SEQ1,
                ),
            }
        }
        self.data.sequences = new_sequences

        self.assertEqual(self.data.sequences, new_sequences)

    def test_add_sequence(self):
        """Test adding a single sequence."""
        new_sequence = Sequence(
            "CGA",
            chain="A",
            index=4,
            active=True,
            parent=None,
            child=None,
            native=SEQ1,
        )
        self.data.add_sequence(chain="A", new_sequence=new_sequence)

        expected_sequences = {
            "A": {
                SEQ1.sequence: SEQ1,
                SEQ2.sequence: SEQ2,
                SEQ3.sequence: SEQ3,
                new_sequence.sequence: new_sequence,
            }
        }
        seqs_data = [seq for seq in self.data.sequences["A"]]
        seqs_expected = [seq for seq in expected_sequences["A"]]
        self.assertEqual(seqs_data, seqs_expected)

    def test_add_existent_sequence(self):
        """Test adding a single sequence."""

        self.data.add_sequence(chain="A", new_sequence=SEQ1)

        expected_sequences = {
            "A": {
                SEQ1.sequence: SEQ1,
                SEQ2.sequence: SEQ2,
                SEQ3.sequence: SEQ3,
            }
        }
        seqs_data = [seq for seq in self.data.sequences["A"]]
        seqs_expected = [seq for seq in expected_sequences["A"]]
        self.assertEqual(seqs_data, seqs_expected)

    def test_get_data(self):
        """Test retrieving data from the singleton."""

        self.data.sequences = deepcopy(SEQUENCES_C)

        sequences = self.data.sequences

        seqs_data = [seq for seq in sequences["A"]]
        seqs_expected = [seq for seq in SEQUENCES_C["A"]]

        self.assertEqual(seqs_data, seqs_expected)

    def test_nsequences(self):
        """Test counting the number of sequences."""
        self.data.sequences = deepcopy(SEQUENCES_C)
        self.assertEqual(self.data.nsequences("A"), 3)

    def test_singleton_instance_created_once(self):
        """Test that the singleton instance is created only once."""
        instance1 = AlgorithmDataSingleton(sequences=deepcopy(SEQUENCES_C))
        instance2 = AlgorithmDataSingleton(sequences={"A": Seq("ATG")})
        self.assertEqual(instance1, instance2)

    def test_multiple_instances_return_same_singleton(self):
        """Test that multiple instances return the same singleton."""
        instance1 = AlgorithmDataSingleton(sequences={"A": Seq("ATG")})
        self.assertEqual(instance1.sequences["A"].keys(), SEQUENCES_C["A"].keys())

    def test_empty_sequences(self):
        """Test behaviour with empty sequences."""
        self.data.sequences = {}
        self.assertEqual(self.data.sequences, {})


if __name__ == "__main__":
    unittest.main()
