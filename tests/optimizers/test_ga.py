import unittest
from unittest.mock import MagicMock

from Bio.Seq import Seq
from icecream import ic

from mood.base.data import AlgorithmDataSingleton
from mood.optimizers.genetic_algorithm import GeneticAlgorithm

# from optimizers.genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.seq_ini = "HNPVVMVHGMGGASYNFASIKSYLVTQGWDRNQLFAIDFIDKTGNNRNNGPRLSRFVKDVLGKTGAKKVDIVAHSMGGANTLYYIKNLDGGDKIENVVTLGGANGLVSLRALPGTDPNQKILYTSVYSSADMIVVNSLSRLIGARNVLIHGVGHISLLASSQVKGYIKEGLNGGGQNTNLE"
        self.chains = ["A"]
        sequences = {chain: {} for chain in self.chains}
        self.data = AlgorithmDataSingleton(sequences=sequences, chains=self.chains)
        self.ga = GeneticAlgorithm(
            data=self.data,
            population_size=10,
            mutable_positions={
                "A": [34, 36, 55, 66, 70, 113, 120, 121, 122, 155, 156, 184]
            },
            mutable_aa={
                "A": {
                    34: ["H", "D", "Q", "S", "K", "M", "F", "L", "T", "A", "C"],
                    36: ["R", "E", "D", "K", "A", "M", "S", "H", "L"],
                    55: ["A", "Q", "E", "M", "L", "V", "C", "D", "K", "S", "F"],
                    66: ["Q", "R", "E", "A", "M", "L", "S", "F", "H"],
                    70: [
                        "Q",
                        "D",
                        "S",
                        "H",
                        "T",
                        "M",
                        "A",
                        "F",
                        "R",
                        "N",
                        "V",
                        "C",
                        "E",
                    ],
                    113: ["E", "D", "H", "Q", "S", "A", "C", "K", "N", "T", "V", "M"],
                    120: ["S", "D", "H", "Q", "T", "C", "G", "A", "R", "K"],
                    121: ["D", "S", "H", "T", "C", "V", "Y", "K", "Q", "F"],
                    122: ["S", "H", "D", "K", "M", "R", "T", "A"],
                    155: ["L", "C", "F", "M", "S", "A", "H", "I", "T"],
                    156: ["H", "S", "T", "C", "D", "V"],
                    184: ["K", "Q", "A", "R"],
                }
            },
            seed=12345,
            debug=True,
        )

    def test_init_population(self):
        sequences_initial = self.seq_ini
        self.ga.data.clear_data()
        self.ga.data.sequences = {chain: {} for chain in self.chains}
        seqs = self.ga.init_population(chain="A", sequences_initial=sequences_initial)
        with open("seqs.txt", "w") as f:
            for seq in seqs:
                f.write(str(seq) + "\n")
        self.assertEqual(len(self.ga.child_sequences), self.ga.population_size)

    # def test_generate_mutation_sequence(self):
    #     sequence = Seq("ATGCKPLQWR")
    #     mutated_seq1, old_aa, new_aa = self.ga.generate_mutation_sequence(sequence, 1.0)
    #     self.assertEqual(mutated_seq1, "PMSGKAQQWN")
    #     self.assertEqual(
    #         old_aa, {1: "A", 2: "T", 3: "G", 4: "C", 6: "P", 7: "L", 10: "R"}
    #     )
    #     self.assertEqual(
    #         new_aa, {1: "P", 2: "M", 3: "S", 4: "G", 6: "A", 7: "Q", 10: "N"}
    #     )
    #     mutated_seq2, old_aa, new_aa = self.ga.generate_mutation_sequence(sequence, 1.0)
    #     self.assertEqual(mutated_seq2, "PMCAKCQQWV")
    #     self.assertEqual(
    #         old_aa, {1: "A", 2: "T", 3: "G", 4: "C", 6: "P", 7: "L", 10: "R"}
    #     )
    #     self.assertEqual(
    #         new_aa, {1: "P", 2: "M", 3: "C", 4: "A", 6: "C", 7: "Q", 10: "V"}
    #     )
    #     self.assertTrue(mutated_seq1[0] in ["S", "P"])
    #     self.assertEqual(mutated_seq1[1], "M")

    # def test_uniform_crossover(self):
    #     seq1 = Seq("AAAAAAAAAA")
    #     seq2 = Seq("TTTTTTTTTT")
    #     crossover_seq = self.ga.uniform_crossover(seq1, seq2)
    #     self.assertEqual(crossover_seq, "TTATTTATTT")
    #     crossover_seq2 = self.ga.uniform_crossover(seq1, seq2)
    #     self.assertEqual(crossover_seq2, "ATATATATAT")
    #     crossover_seq3 = self.ga.uniform_crossover(seq1, seq2)
    #     self.assertEqual(crossover_seq3, "ATTTTAAATA")

    # # Performing two-point crossover correctly
    # def test_two_point_crossover(self):
    #     seq1 = Seq("RRRRRRRRRR")
    #     seq2 = Seq("VVVVVVVVVV")
    #     crossover_seq1 = self.ga.two_point_crossover(seq1, seq2, start=6, end=10)
    #     self.assertEqual(crossover_seq1, "RRRRRVVVVV")
    #     crossover_seq2 = self.ga.two_point_crossover(seq1, seq2, start=5, end=5)
    #     self.assertEqual(crossover_seq2, "RRRRVRRRRR")
    #     crossover_seq3 = self.ga.two_point_crossover(seq1, seq2, start=1, end=10)
    #     self.assertEqual(crossover_seq3, "VVVVVVVVVV")
    #     crossover_seq4 = self.ga.two_point_crossover(seq1, seq2, start=11, end=10)
    #     self.assertEqual(crossover_seq4, "RRRRRRRRRR")

    # # Performing single-point crossover correctly
    # def test_single_point_crossover(self):
    #     seq1 = Seq("IIIIIIIIII")
    #     seq2 = Seq("LLLLLLLLLL")
    #     crossover_seq = self.ga.single_point_crossover(seq1, seq2, crossover_point=6)
    #     self.assertEqual(crossover_seq, "IIIIILLLLL")
    #     crossover_seq2 = self.ga.single_point_crossover(seq1, seq2, crossover_point=1)
    #     self.assertEqual(crossover_seq2, "LLLLLLLLLL")
    #     crossover_seq3 = self.ga.single_point_crossover(seq1, seq2, crossover_point=11)
    #     self.assertEqual(crossover_seq3, "IIIIIIIIII")

    # # Handling invalid crossover type gracefully
    # def test_invalid_crossover_type(self):
    #     with self.assertRaises(ValueError) as context:
    #         self.ga.generate_crossover_sequence(
    #             Seq("AAAA"), Seq("TTTT"), crossover_type="invalid"
    #         )
    #     self.assertIn("Invalid crossover type", str(context.exception))

    # # # Initializing population with zero initial sequences
    # # def test_empty_initial_sequences_list(self):
    # #     sequences_initial = []
    # #     with self.assertRaises(ValueError) as context:
    # #         self.ga.init_population(sequences_initial)
    # #     self.assertIn("No initial sequences provided", str(context.exception))
    # #     self.assertEqual(len(self.sequences), self.ga.population_size)

    # # Handling empty mutable positions list
    # def test_empty_mutable_positions(self):
    #     ga = GeneticAlgorithm(
    #         mutable_positions=[], mutable_aa={1: ["A", "P", "S"], 2: ["M"]}
    #     )
    #     sequence = Seq("ATGC")
    #     with self.assertRaises(ValueError) as context:
    #         _, _, _ = ga.generate_mutation_sequence(sequence, 1.0)
    #     self.assertIn("No mutable positions provided", str(context.exception))

    # # Handling empty mutable amino acids dictionary
    # def test_empty_mutable_aa_dict(self):
    #     ga = GeneticAlgorithm(mutable_positions=[1, 2], mutable_aa={})
    #     sequence = Seq("ATGC")
    #     with self.assertRaises(ValueError) as context:
    #         _, _, _ = ga.generate_mutation_sequence(sequence, 1.0)
    #     self.assertIn("No mutable amino acids provided", str(context.exception))

    # # Handling mutation rate of zero
    # def test_mutation_rate_zero(self):
    #     sequence = Seq("ATGC")
    #     mutated_seq, old_aa, new_aa = self.ga.generate_mutation_sequence(sequence, 0.0)
    #     self.assertEqual(mutated_seq, sequence)
    #     self.assertEqual(old_aa, {})
    #     self.assertEqual(new_aa, {})

    # def check_duplicate_sequences(self):
    #     sequence_counts = {}
    #     # Count each sequence's occurrences
    #     for seq in self.ga.data.sequences:
    #         if seq in sequence_counts:
    #             sequence_counts[seq] += 1
    #         else:
    #             sequence_counts[seq] = 1
    #     # Check for any sequence that appears more than once
    #     for count in sequence_counts.values():
    #         if count > 1:
    #             return True
    #     return False

    # def test_generate_child_population(self):
    #     sequences_initial = [Seq("ATGCKPLQWR")]
    #     seqs = self.ga.init_population(sequences_initial)

    #     self.assertEqual(len(seqs), self.ga.population_size)
    #     # save seqs in a file
    #     with open("seqs.txt", "w") as f:
    #         for seq in seqs:
    #             f.write(str(seq) + "\n")
    #     print("saved")
    #     childs = self.ga.generate_child_population(seqs)
    #     self.assertEqual(len(childs), self.ga.population_size)
    #     duplicate_found = self.check_duplicate_sequences()
    #     self.assertFalse(
    #         duplicate_found,
    #         "No sequence appears more than once in self.ga.data.sequences",
    #     )

    # def test_eval_population(self):
    #     import pandas as pd

    #     data_frame = pd.DataFrame(
    #         {
    #             "seq_index": [
    #                 0,
    #                 1,
    #                 2,
    #                 3,
    #             ],
    #             "Sequence": [
    #                 Seq("ATG"),
    #                 Seq("TGA"),
    #                 Seq("GAT"),
    #                 Seq("PIU"),
    #             ],
    #             "iteration": [
    #                 1,
    #                 1,
    #                 2,
    #                 2,
    #             ],
    #             "Metric1": [-2, -6, -1, -1],
    #             "Metric2": [-1, -6, -1, -3],
    #         }
    #     )

    #     ranked_df = self.ga.eval_population(data_frame)

    #     # Assert that the Rank column has the expected values
    #     expected_ranks = [2.0, 1.0, 2.0, 2.0]
    #     assert (
    #         ranked_df["Rank"].tolist() == expected_ranks
    #     ), f"Expected ranks {expected_ranks}, but got {ranked_df['Rank'].tolist()}"

    # def test_eval_population(self):
    #     import pandas as pd

    #     data_frame = pd.DataFrame(
    #         {
    #             "Sequence": [
    #                 Seq("ATG"),
    #                 Seq("TGA"),
    #                 Seq("GAT"),
    #                 Seq("PIU"),
    #             ],
    #             "Metric1": [2, 1, 3, 3],
    #             "Metric2": [-3, -1, -3, -1],
    #         }
    #     )
    #     """
    #     "Metric1": [2, 6, 1, 1],
    #     "Metric2": [1, 6, 1, 3],

    #     "Metric1": [-2, -6, -1, -1],
    #     "Metric2": [-1, -6, -1, -3],

    #     "Metric1": [2, 1, 3, 3],
    #     "Metric2": [-3, -1, -3, -1],
    #     """

    #     states = {"Metric1": "Positive", "Metric2": "Negative"}

    #     ranked_df = self.ga.calculate_non_dominated_rank(data_frame, states)

    #     # Assert that the Rank column has the expected values
    #     expected_ranks = [2.0, 1.0, 2.0, 2.0]
    #     assert (
    #         ranked_df["Rank"].tolist() == expected_ranks
    #     ), f"Expected ranks {expected_ranks}, but got {ranked_df['Rank'].tolist()}"


# Usage example


if __name__ == "__main__":
    unittest.main()
