import json
import os
import shutil
import unittest

from mood.base.data import AlgorithmDataSingleton
from mood.metrics.proteinMPNN_Metrics import ProteinMPNNMetrics
from mood.metrics.pyrosetta_Metrics import RosettaMetrics
from mood.multiObjectiveOptimization import MultiObjectiveOptimization


class TestmultiObjectiveOptimization(unittest.TestCase):
    def setUp(self):
        data = AlgorithmDataSingleton()
        optimizer = "genetic_algorithm"
        params_folder = "tests/data/Felip9/PET_params"
        cpus = 4
        native_pdb = "tests/data/Felip9/FeLip9-PET-1.pdb"
        distances_file = "tests/data/distances.json"
        cst_file = "tests/data/Felip9/FeLip9-PET-1_CA.cst"
        metrics = [
            RosettaMetrics(
                params_folder=params_folder,
                cpus=cpus,
                seed=1235,
                native_pdb=native_pdb,
                distances_file=distances_file,
                cst_file=cst_file,
            ),
            ProteinMPNNMetrics(chain="A", seed=1235, native_pdb=native_pdb),
        ]
        debug = True
        max_iteration = 50
        population_size = 20
        seed = 1235

        chains = ["A"]
        # mutable_positions = {
        #     "A": [34, 36, 55, 66, 70, 113, 120, 121, 122, 155, 156, 184]
        # }
        with open(
            "tests/data/Felip9/design_library_glide.json",
            "r",
        ) as f:
            design = json.load(f)
        mutable_aa = {"A": design}
        folder_name = "mood_job"
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        self.moo = MultiObjectiveOptimization(
            optimizer=optimizer,
            metrics=metrics,
            debug=debug,
            max_iteration=max_iteration,
            native_pdb=native_pdb,
            chains=chains,
            data=data,
            mutable_aa=mutable_aa,
            folder_name=folder_name,
            seed=seed,
            population_size=population_size,
            offset=1,
        )
        if os.path.exists("mood_job"):
            shutil.rmtree("mood_job")

    def test_run(self):
        self.moo.run()
        self.assertTrue(os.path.exists("mood_job/000"))


if __name__ == "__main__":
    unittest.main()
