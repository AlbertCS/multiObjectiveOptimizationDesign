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
        data_path = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/hmfo"

        data = AlgorithmDataSingleton()
        optimizer = "genetic_algorithm"
        params = f"{data_path}/params"
        cpus = 4
        native_pdb = f"{data_path}/AF-A0A9E4RQT2@HFFCA@146.pdb"
        pdb_scafold = f"{data_path}/A0A9E4RQT2.pdb"
        fixed_positions = f"{data_path}/pocket_residues.json"

        metrics = [
            RosettaMetrics(
                params_folder=params,
                cpus=cpus,
                seed=1235,
                native_pdb=native_pdb,
                ligand_chain="L",
            ),
            ProteinMPNNMetrics(
                seed=1235, native_pdb=pdb_scafold, fixed_positions=fixed_positions
            ),
        ]
        debug = True
        max_iteration = 50
        population_size = 10
        seed = 1235
        # eval_mutations_params = {
        #     "min_energy_threshold": 0,
        #     "seed": seed,
        #     "params_folder": params,
        #     "native_pdb": native_pdb,
        # }
        chains = ["A"]

        # Folder name and remove if exists
        folder_name = "mood_job"
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)

        self.moo = MultiObjectiveOptimization(
            optimizer=optimizer,
            metrics=metrics,
            debug=debug,
            max_iteration=max_iteration,
            native_pdb=native_pdb,
            pdb_scafold=pdb_scafold,
            chains=chains,
            data=data,
            folder_name=folder_name,
            seed=seed,
            population_size=population_size,
            offset=2,
            fixed_positions=fixed_positions,
            # eval_mutations_params=eval_mutations_params,
            mutation_probability=True,
        )

    def test_run(self):
        self.moo.run()
        self.assertTrue(os.path.exists("mood_job/000"))


if __name__ == "__main__":
    unittest.main()
