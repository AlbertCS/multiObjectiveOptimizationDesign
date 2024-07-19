import unittest
from unittest.mock import MagicMock

from Bio.Seq import Seq
from pytest_mock import mocker

from mood.optimizers.genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    # def setUp(self):
    #     self.genetic_algorithm = GeneticAlgorithm(population_size=5)

    # def test_init_population(self):
    #     # Mock the data object
    #     data_mock = mocker.Mock()
    #     ga = GeneticAlgorithm(data=data_mock)
    #     sequences_initial = [Seq("ATGC")]
    #     ga.init_population(sequences_initial)
    #     self.assertEqual(len(data_mock.add_sequence.call_args_list), ga.population_size)

    def test_generate_mutation_sequence(self):
        ga = GeneticAlgorithm(
            mutable_positions=[1, 2], mutable_aa={1: ["A", "P", "S"], 2: ["M"]}
        )
        sequence = Seq("ATGC")
        mutated_seq, old_aa, new_aa = ga.generate_mutation_sequence(sequence, 1.0)
        self.assertNotEqual(mutated_seq, sequence)
        self.assertIn(1, old_aa)
        self.assertIn(2, old_aa)
        self.assertEqual(mutated_seq[1], "M")

    # def test_uniform_crossover(self):
    #     ga = GeneticAlgorithm(mutable_positions=[1, 2])
    #     seq1 = Seq("AAAA")
    #     seq2 = Seq("TTTT")
    #     crossover_seq = ga.uniform_crossover(seq1, seq2)
    #     self.assertNotEqual(crossover_seq, seq1)
    #     self.assertNotEqual(crossover_seq, seq2)

    # Performing two-point crossover correctly
    # def test_two_point_crossover(self):
    #     ga = GeneticAlgorithm(mutable_positions=[1, 2])
    #     seq1 = Seq("AAAA")
    #     seq2 = Seq("TTTT")
    #     crossover_seq = ga.two_point_crossover(seq1, seq2)
    #     self.assertNotEqual(crossover_seq, seq1)
    #     self.assertNotEqual(crossover_seq, seq2)

    # # Performing single-point crossover correctly
    # def test_single_point_crossover(self):
    #     ga = GeneticAlgorithm(mutable_positions=[1, 2])
    #     seq1 = Seq("AAAA")
    #     seq2 = Seq("TTTT")
    #     crossover_seq = ga.single_point_crossover(seq1, seq2)
    #     self.assertNotEqual(crossover_seq, seq1)
    #     self.assertNotEqual(crossover_seq, seq2)

    # # Initializing population with zero initial sequences
    # def test_init_population_zero_initial_sequences(self):
    #     data_mock = mocker.Mock()
    #     ga = GeneticAlgorithm(data=data_mock)
    #     sequences_initial = []
    #     ga.init_population(sequences_initial)
    #     self.assertEqual(len(data_mock.add_sequence.call_args_list), ga.population_size)

    # # Handling invalid crossover type gracefully
    # def test_invalid_crossover_type(self):
    #     ga = GeneticAlgorithm()
    #     with self.assertRaises(Exception) as context:
    #         ga.generate_crossover_sequence(
    #             Seq("AAAA"), Seq("TTTT"), crossover_type="invalid"
    #         )
    #     self.assertIn("Invalid crossover type", str(context.exception))

    # # Handling empty mutable positions list
    # def test_empty_mutable_positions(self):
    #     ga = GeneticAlgorithm(mutable_positions=[])
    #     sequence = Seq("ATGC")
    #     mutated_seq, old_aa, new_aa = ga.generate_mutation_sequence(sequence, 1.0)
    #     self.assertEqual(mutated_seq, sequence)
    #     self.assertEqual(old_aa, {})
    #     self.assertEqual(new_aa, {})

    # # Handling empty mutable amino acids dictionary
    # def test_empty_mutable_aa_dict(self):
    #     ga = GeneticAlgorithm(mutable_aa={})
    #     sequence = Seq("ATGC")
    #     mutated_seq, old_aa, new_aa = ga.generate_mutation_sequence(sequence, 1.0)
    #     self.assertEqual(mutated_seq, sequence)
    #     self.assertEqual(old_aa, {})
    #     self.assertEqual(new_aa, {})

    # # Handling mutation rate of zero
    # def test_mutation_rate_zero(self):
    #     ga = GeneticAlgorithm(mutable_positions=[1, 2], mutable_aa={1: "A", 2: "T"})
    #     sequence = Seq("ATGC")
    #     mutated_seq, old_aa, new_aa = ga.generate_mutation_sequence(sequence, 0.0)
    #     self.assertEqual(mutated_seq, sequence)
    #     self.assertEqual(old_aa, {})
    #     self.assertEqual(new_aa, {})

    # def test_empty_initial_sequences_list(self):
    #     data_mock = mocker.Mock()
    #     ga = GeneticAlgorithm(data=data_mock)
    #     sequences_initial = []
    #     ga.init_population(sequences_initial)
    #     self.assertEqual(len(data_mock.add_sequence.call_args_list), ga.population_size)


if __name__ == "__main__":
    unittest.main()
