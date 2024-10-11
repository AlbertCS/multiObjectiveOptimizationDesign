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
        eval_mutations_params = {
            "cst_file": "tests/data/Felip9/FeLip9-PET-1_CA.cst",
            "min_energy_threshold": 1,
            "seed": 12345,
            "params_folder": "tests/data/Felip9/PET_params",
            "native_pdb": "tests/data/Felip9/FeLip9-PET-1.pdb",
        }
        self.ga = GeneticAlgorithm(
            data=self.data,
            population_size=10,
            # mutable_positions={
            #     "A": [34, 36, 55, 66, 70, 113, 120, 121, 122, 155, 156, 184]
            # },
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
                    # 184: ["K", "Q", "A", "R"],
                }
            },
            seed=12345,
            debug=True,
            crossover_iterations=1,
            mutation_iterations=1,
            max_mutation_per_iteration=1,
            eval_mutations=True,
            eval_mutations_params=eval_mutations_params,
        )

    def test_init_population(self):
        sequences_initial = self.seq_ini
        self.ga.data.sequences = {chain: {} for chain in self.chains}
        seqs = self.ga.init_population(chain="A", sequences_initial=sequences_initial)
        with open("seqs.txt", "w") as f:
            for seq in seqs:
                f.write(str(seq) + "\n")
        self.assertEqual(len(self.ga.child_sequences), self.ga.population_size)

    def test_generate_mutation_sequence(self):
        sequence = "ATGCKPLQWR"
        self.ga.mutable_aa = {
            "A": {
                1: ["S", "P"],
                2: ["M"],
                3: ["S", "C"],
                4: ["G", "A"],
                6: ["A", "P", "C"],
                7: ["Q", "L"],
                10: ["N"],
            }
        }
        mutated_seq1, mut = self.ga.generate_mutation_sequence(
            sequence_to_mutate=sequence, chain="A"
        )
        self.assertEqual(mutated_seq1, "ATGCKPLQWR")
        self.assertEqual(
            mut,
            [
                ("Q", 7, "Q"),
            ],
        )
        mutated_seq2, mut = self.ga.generate_mutation_sequence(
            sequence_to_mutate=sequence, chain="A"
        )
        self.assertEqual(mutated_seq2, "ATGCKPLQWN")
        self.assertEqual(
            mut,
            [
                ("R", 10, "N"),
            ],
        )

    def test_uniform_crossover(self):
        seq1 = "AAAAAAAAAA"
        seq2 = "TTTTTTTTTT"
        self.ga.mutable_aa = {
            "A": {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
        }
        crossover_seq = self.ga.uniform_crossover(seq1, seq2, chain="A")
        self.assertEqual(crossover_seq, "ATATATATTA")
        crossover_seq2 = self.ga.uniform_crossover(seq1, seq2, chain="A")
        self.assertEqual(crossover_seq2, "ATAAATAAAT")
        crossover_seq3 = self.ga.uniform_crossover(seq1, seq2, chain="A")
        self.assertEqual(crossover_seq3, "ATATAAAATA")

    # Performing two-point crossover correctly
    def test_two_point_crossover(self):
        seq1 = "RRRRRRRRRR"
        seq2 = "VVVVVVVVVV"
        self.ga.mutable_aa = {
            "A": {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
        }
        crossover_seq1 = self.ga.two_point_crossover(
            seq1, seq2, start=6, end=10, chain="A"
        )
        self.assertEqual(crossover_seq1, "RRRRRVVVVV")
        crossover_seq2 = self.ga.two_point_crossover(
            seq1, seq2, start=5, end=5, chain="A"
        )
        self.assertEqual(crossover_seq2, "RRRRVRRRRR")
        crossover_seq3 = self.ga.two_point_crossover(
            seq1, seq2, start=1, end=10, chain="A"
        )
        self.assertEqual(crossover_seq3, "VVVVVVVVVV")
        crossover_seq4 = self.ga.two_point_crossover(
            seq1, seq2, start=11, end=10, chain="A"
        )
        self.assertEqual(crossover_seq4, "RRRRRRRRRR")

    # Performing single-point crossover correctly
    def test_single_point_crossover(self):
        seq1 = "IIIIIIIIII"
        seq2 = "LLLLLLLLLL"
        self.ga.mutable_aa = {
            "A": {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
        }
        crossover_seq = self.ga.single_point_crossover(
            seq1, seq2, crossover_point=6, chain="A"
        )
        self.assertEqual(crossover_seq, "IIIIILLLLL")
        crossover_seq2 = self.ga.single_point_crossover(
            seq1, seq2, crossover_point=1, chain="A"
        )
        self.assertEqual(crossover_seq2, "LLLLLLLLLL")
        crossover_seq3 = self.ga.single_point_crossover(
            seq1, seq2, crossover_point=11, chain="A"
        )
        self.assertEqual(crossover_seq3, "IIIIIIIIII")

    # Handling invalid crossover type gracefully
    def test_invalid_crossover_type(self):
        with self.assertRaises(ValueError) as context:
            self.ga.generate_crossover_sequence(
                "AAAA", "TTTT", crossover_type="invalid"
            )
        self.assertIn("Invalid crossover type", str(context.exception))

    # Handling empty mutable amino acids dictionary
    def test_empty_mutable_aa_dict(self):
        ga = GeneticAlgorithm(mutable_aa={})
        sequence = "ATGC"
        with self.assertRaises(ValueError) as context:
            _, _ = ga.generate_mutation_sequence(sequence_to_mutate=sequence, chain="A")
        self.assertIn("No mutable amino acids provided", str(context.exception))

    def check_duplicate_sequences(self):
        sequence_counts = {}
        # Count each sequence's occurrences
        for seq in self.ga.data.sequences:
            if seq in sequence_counts:
                sequence_counts[seq] += 1
            else:
                sequence_counts[seq] = 1
        # Check for any sequence that appears more than once
        for count in sequence_counts.values():
            if count > 1:
                return True
        return False

    def test_generate_child_population(self):

        self.setUp()
        sequences_initial = self.seq_ini
        seqs = self.ga.init_population(chain="A", sequences_initial=sequences_initial)
        self.assertEqual(len(seqs), self.ga.population_size)
        # save seqs in a file
        # with open("seqs.txt", "w") as f:
        #     for seq in seqs:
        #         f.write(str(seq) + "\n")
        childs = self.ga.generate_child_population(seqs, chain="A", current_iteration=1)
        self.assertEqual(len(childs), self.ga.population_size)
        duplicate_found = self.check_duplicate_sequences()
        self.assertFalse(
            duplicate_found,
            "No sequence appears more than once in self.ga.data.sequences",
        )

    def test_evaluate_mutation(self):
        sequence = "EHNPVVMVHGMGGASYNFASIKSYLVTQGWDRNQLFAIDFIDKTGNNRNNGPRLSRMVKDVLGKTGAKKVDIVAHSMGGANTLYYIKNLGGGDKIENVVTLGGANGLVSLRALPGTDPNQKGSYTSVYSSADMIVVNSLSRLIGARNVLIHGVGHISLLASSQVKGYIKEGLNGGGQNTN"
        mutated_seq = "PHNPVVMVHGMGGASYNFASIKSYLVTQGWDRNQLFAIDFIDKTGNNRNNGPRLSRMVKDVLGKTGAKKVDIVAHSMGGANTLYYIKNLGGGDKIENVVTLGGANGLVSLRALPGTDPNQKGSYTSVYSSADMIVVNSLSRLIGARNVLIHGVGHISLLASSQVKGYIKEGLNGGGQNTN"
        # rs = mut[0][1]
        # Rosetta starts at index 1
        dEnergy = self.ga.local_relax(
            residues=[1],
            moving_chain="A",
            starting_sequence=sequence,
            mutated_sequence=mutated_seq,
            cst_file=self.ga.eval_mutations_params["cst_file"],
        )
        if dEnergy < self.ga.eval_mutations_params["min_energy_threshold"]:
            print("Mutation with worst energy")


# Usage example


if __name__ == "__main__":
    unittest.main()
