import os
import shutil
import unittest

from Bio.Seq import Seq
from icecream import ic

from mood.base.data import AlgorithmDataSingleton
from mood.metrics.pyrosettaMetrics import RosettaMetrics
from mood.multiObjectiveOptimization import MultiObjectiveOptimization


class TestmultiObjectiveOptimization(unittest.TestCase):
    def setUp(self):
        # TODO send all the data to the repo
        data = AlgorithmDataSingleton()
        optimizer = "genetic_algorithm"
        params_folder = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/Felip9/PET_params"
        cpus = 6
        native_pdb = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/Felip9/FeLip9-PET-1.pdb"
        distances_file = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/distances.json"
        cst_file = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/Felip9/FeLip9-PET-1_CA.cst"
        metrics = [
            RosettaMetrics(
                params_folder=params_folder,
                cpus=cpus,
                seed=1235,
                native_pdb=native_pdb,
                distances_file=distances_file,
                cst_file=cst_file,
            )
        ]
        debug = True
        max_iteration = 5
        population_size = 20
        seed = 1235

        chains = ["A"]
        # mutable_positions = {
        #     "A": [34, 36, 55, 66, 70, 113, 120, 121, 122, 155, 156, 184]
        # }
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
            chains=chains,
            data=data,
            mutable_aa=mutable_aa,
            folder_name=folder_name,
            seed=seed,
            population_size=population_size,
            offset=3,
        )
        if os.path.exists("mood_job"):
            shutil.rmtree("mood_job")

    def test_run(self):
        self.moo.run()
        self.assertTrue(os.path.exists("mood_job/000"))


if __name__ == "__main__":
    unittest.main()
