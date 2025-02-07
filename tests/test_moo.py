import os
import pickle
import shutil
import unittest
from unittest.mock import MagicMock

from Bio.Seq import Seq

from mood.base.data import AlgorithmDataSingleton
from mood.metrics.alphabetic import Alphabet
from mood.multiObjectiveOptimization import MultiObjectiveOptimization


class TestmultiObjectiveOptimization(unittest.TestCase):
    def setUp(self):
        data = AlgorithmDataSingleton()
        optimizer = "genetic_algorithm"
        metrics = [Alphabet()]
        debug = True
        max_iteration = 5
        population_size = 20
        seed = 1235
        pdb = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/7R1K.pdb"
        chains = ["A"]
        mutable_positions = {
            "A": [34, 36, 55, 66, 70, 113, 120, 121, 122, 155, 156, 184]
        }
        mutable_aa = {
            "A": {
                34: ["H", "D", "Q", "S", "K", "M", "F", "L", "T", "A", "C"],
                36: ["R", "E", "D", "K", "A", "M", "S", "H", "L"],
                55: ["A", "Q", "E", "M", "L", "V", "C", "D", "K", "S", "F"],
                66: ["Q", "R", "E", "A", "M", "L", "S", "F", "H"],
                70: ["Q", "D", "S", "H", "T", "M", "A", "F", "R", "N", "V", "C", "E"],
                113: ["E", "D", "H", "Q", "S", "A", "C", "K", "N", "T", "V", "M"],
                120: ["S", "D", "H", "Q", "T", "C", "G", "A", "R", "K"],
                121: ["D", "S", "H", "T", "C", "V", "Y", "K", "Q", "F"],
                122: ["S", "H", "D", "K", "M", "R", "T", "A"],
                155: ["L", "C", "F", "M", "S", "A", "H", "I", "T"],
                156: ["H", "S", "T", "C", "D", "V"],
                184: ["K", "Q", "A", "R"],
            }
        }
        folder_name = "mood_job"
        self.moo = MultiObjectiveOptimization(
            optimizer=optimizer,
            metrics=metrics,
            debug=debug,
            max_iteration=max_iteration,
            pdb=pdb,
            chains=chains,
            data=data,
            mutable_positions=mutable_positions,
            mutable_aa=mutable_aa,
            folder_name=folder_name,
            seed=seed,
            population_size=population_size,
            offset=3,
        )
        if os.path.exists("mood_job"):
            shutil.rmtree("mood_job")

    def test_setup_folders(self):
        self.moo.setup_folders_initial()
        self.assertTrue(self.moo.folder_name in os.listdir())

    def test_get_seq_from_pdb(self):
        seq = self.moo._get_seq_from_pdb(
            pdb_file="/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/7R1K.pdb"
        )
        se = "HNPVVMVHGMGGASYNFASIKSYLVTQGWDRNQLFAIDFIDKTGNNRNNGPRLSRFVKDVLGKTGAKKVDIVAHSMGGANTLYYIKNLDGGDKIENVVTLGGANGLVSLRALPGTDPNQKILYTSVYSSADMIVVNSLSRLIGARNVLIHGVGHISLLASSQVKGYIKEGLNGGGQNTNLE"
        self.assertEqual(seq["A"], se)

    def test_check_previous_iterations_none(self):
        self.moo.check_previous_iterations()
        self.assertTrue(self.moo.data.iteration == 0)

    def test_check_iter_finished(self):
        sequences = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/sequences.pkl"
        dataframe = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/data_frame.pkl"
        finished = self.moo.check_Iter_finished(
            iteration=0, sequences_pkl=sequences, dataframe_pkl=dataframe
        )
        self.assertEqual(finished, True)

    def test_check_previous_iterations(self):
        if os.path.exists("mood_job/000"):
            shutil.rmtree("mood_job/000")
        else:
            os.mkdir("mood_job")
        os.mkdir("mood_job/000")
        shutil.copy(
            "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/sequences.pkl",
            "mood_job/000/sequences.pkl",
        )
        shutil.copy(
            "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/data_frame.pkl",
            "mood_job/000/data_frame.pkl",
        )
        finished, dataframe = self.moo.check_previous_iterations()
        self.assertTrue(self.moo.current_iteration == 1)

    def test_check_previous_iterations_none(self):
        if not os.path.exists("mood_job"):
            os.mkdir("mood_job")
        if os.path.exists("mood_job/000"):
            shutil.rmtree("mood_job/000")

        _, _ = self.moo.check_previous_iterations()
        self.assertTrue(self.moo.current_iteration == 0)

    def test_save_iteration(self):
        sequences = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/sequences.pkl"
        dataframe = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/data_frame.pkl"
        current_iteration = 1
        with open(sequences, "rb") as f:
            sequences = pickle.load(f)
        with open(dataframe, "rb") as f:
            dataframe = pickle.load(f)
        if not os.path.exists("mood_job"):
            os.mkdir("mood_job")
        if not os.path.exists("mood_job/001"):
            os.mkdir("mood_job/001")
        self.moo.save_iteration(current_iteration, sequences, dataframe)
        seq1 = "mood_job/001/sequences.pkl"
        df1 = "mood_job/001/data_frame.pkl"
        self.assertTrue(os.path.exists(seq1))
        self.assertTrue(os.path.exists(df1))

    def test_setup_folders_iter(self):
        if not os.path.exists("mood_job"):
            os.mkdir("mood_job")
        self.moo.setup_folders_iter(current_iteration=2)
        self.assertTrue(os.path.exists("mood_job/002"))

    def test_save_load_info(self):
        if not os.path.exists("mood_job"):
            os.mkdir("mood_job")
        if not os.path.exists("mood_job/input"):
            os.mkdir("mood_job/input")
        seq_native = "HNPVVMVHGMGGASYNFASIKSYLVTQGWDRNQLFAIDFIDKTGNNRNNGPRLSRFVKDVLGKTGAKKVDIVAHSMGGANTLYYIKNLDGGDKIENVVTLGGANGLVSLRALPGTDPNQKILYTSVYSSADMIVVNSLSRLIGARNVLIHGVGHISLLASSQVKGYIKEGLNGGGQNTNLE"

        self.moo.save_info(seq=seq_native)
        self.assertTrue(os.path.exists("mood_job/input/info.pkl"))

        self.moo.load_info()
        self.assertEqual(self.moo.native_sequence, seq_native)

    def test_run(self):
        self.moo.run()
        self.assertTrue(os.path.exists("mood_job/000"))


if __name__ == "__main__":
    unittest.main()
