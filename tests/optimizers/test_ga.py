import unittest
from unittest.mock import MagicMock

from Bio.Seq import Seq
from icecream import ic
from pytest_mock import mocker

from mood.optimizers.genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.ga = GeneticAlgorithm(
            population_size=5,
            mutable_positions=[1, 2, 3, 4, 6, 7, 10],
            mutable_aa={
                1: ["A", "P", "S"],
                2: ["M"],
                3: ["T", "S", "A", "C"],
                4: ["A", "G", "F"],
                6: ["A", "G", "T", "S", "A", "C"],
                7: ["W", "R", "Q", "K", "P", "L"],
                10: ["L", "N", "H", "V"],
            },
        )

    # def test_init_population(self):
    #     sequences_initial = [Seq("ATGCKPLQWR")]
    #     self.ga.init_population(sequences_initial)
    #     self.assertEqual(len(self.ga.sequences), self.ga.population_size)

    def test_generate_mutation_sequence(self):
        sequence = Seq("ATGCKPLQWR")
        print("  **Generate mutation sequence**")
        mutated_seq, old_aa, new_aa = self.ga.generate_mutation_sequence(sequence, 1.0)
        ic(mutated_seq, old_aa, new_aa)
        mutated_seq, old_aa, new_aa = self.ga.generate_mutation_sequence(sequence, 1.0)
        ic(mutated_seq, old_aa, new_aa)
        mutated_seq, old_aa, new_aa = self.ga.generate_mutation_sequence(sequence, 1.0)
        ic(mutated_seq, old_aa, new_aa)
        print("\t** * **\n")
        self.assertNotEqual(mutated_seq, sequence)
        self.assertIn(1, old_aa)
        self.assertIn(2, old_aa)
        self.assertTrue(mutated_seq[0] in ["S", "P"])
        self.assertEqual(mutated_seq[1], "M")

    def test_uniform_crossover(self):
        seq1 = Seq("AAAA")
        seq2 = Seq("TTTT")
        print("  **Uniform crossover**")
        crossover_seq = self.ga.uniform_crossover(seq1, seq2)
        ic(crossover_seq)
        crossover_seq = self.ga.uniform_crossover(seq1, seq2)
        ic(crossover_seq)
        crossover_seq = self.ga.uniform_crossover(seq1, seq2)
        ic(crossover_seq)
        print("\t** * **\n")
        self.assertNotEqual(crossover_seq, seq1)
        self.assertNotEqual(crossover_seq, seq2)

    # Performing two-point crossover correctly
    def test_two_point_crossover(self):
        print("  **Two point crossover**")
        seq1 = Seq("RRRRRRRRRRRRRRR")
        seq2 = Seq("VVVVVVVVVVVVVVV")
        crossover_seq = self.ga.two_point_crossover(seq1, seq2)
        ic(crossover_seq)
        crossover_seq = self.ga.two_point_crossover(seq1, seq2)
        ic(crossover_seq)
        crossover_seq = self.ga.two_point_crossover(seq1, seq2)
        ic(crossover_seq)
        print("\t** * **\n")
        self.assertNotEqual(crossover_seq, seq1)
        self.assertNotEqual(crossover_seq, seq2)

    # # Performing single-point crossover correctly
    # def test_single_point_crossover(self):
    #     seq1 = Seq("AAAA")
    #     seq2 = Seq("TTTT")
    #     crossover_seq = self.ga.single_point_crossover(seq1, seq2)
    #     self.assertNotEqual(crossover_seq, seq1)
    #     self.assertNotEqual(crossover_seq, seq2)

    # # Initializing population with zero initial sequences
    # def test_empty_initial_sequences_list(self):
    #     sequences_initial = []
    #     with self.assertRaises(ValueError) as context:
    #         self.ga.init_population(sequences_initial)
    #     self.assertIn("No initial sequences provided", str(context.exception))
    #     self.assertEqual(len(self.sequences), self.ga.population_size)

    # # Handling invalid crossover type gracefully
    # def test_invalid_crossover_type(self):
    #     with self.assertRaises(ValueError) as context:
    #         self.ga.generate_crossover_sequence(
    #             Seq("AAAA"), Seq("TTTT"), crossover_type="invalid"
    #         )
    #     self.assertIn("Invalid crossover type", str(context.exception))

    # # Handling empty mutable positions list
    # def test_empty_mutable_positions(self):
    #     ga = GeneticAlgorithm(
    #         mutable_positions=[], mutable_aa={1: ["A", "P", "S"], 2: ["M"]}
    #     )
    #     sequence = Seq("ATGC")
    #     with self.assertRaises(ValueError) as context:
    #         mutated_seq, old_aa, new_aa = ga.generate_mutation_sequence(sequence, 1.0)
    #     self.assertIn("No mutable positions provided", str(context.exception))
    #     self.assertEqual(mutated_seq, sequence)
    #     self.assertEqual(old_aa, {})
    #     self.assertEqual(new_aa, {})

    # # Handling empty mutable amino acids dictionary
    # def test_empty_mutable_aa_dict(self):
    #     ga = GeneticAlgorithm(mutable_positions=[1, 2], mutable_aa={})
    #     sequence = Seq("ATGC")
    #     with self.assertRaises(ValueError) as context:
    #         mutated_seq, old_aa, new_aa = ga.generate_mutation_sequence(sequence, 1.0)
    #     self.assertIn("No mutable amino acids provided", str(context.exception))
    #     self.assertEqual(mutated_seq, sequence)
    #     self.assertEqual(old_aa, {})
    #     self.assertEqual(new_aa, {})

    # # Handling mutation rate of zero
    # def test_mutation_rate_zero(self):
    #     sequence = Seq("ATGC")
    #     mutated_seq, old_aa, new_aa = self.ga.generate_mutation_sequence(sequence, 0.0)
    #     self.assertEqual(mutated_seq, sequence)
    #     self.assertEqual(old_aa, {})
    #     self.assertEqual(new_aa, {})


if __name__ == "__main__":
    unittest.main()
