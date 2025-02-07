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
        data_path = "/home/lavane/Users/acanella/PhD/runs/tests_mood/data/FeLip9"

        data = AlgorithmDataSingleton()
        optimizer = "genetic_algorithm"
        params_folder = f"{data_path}/PET_params"
        cpus = 4
        native_pdb = f"{data_path}/FeLip9-PET-1.pdb"
        distances_file = f"{data_path}/distances.json"
        cst_file = f"{data_path}/FeLip9-PET-1_CA.cst"
        starting_sequences = f"{data_path}/ProtNPMM_native_small.fasta"

        # starting_sequences = "tests/data/Felip9/ProtNPMM_native_small.fasta"
        metrics = [
            RosettaMetrics(
                params_folder=None,
                cpus=cpus,
                seed=1235,
                native_pdb=native_pdb,
                ligand_chain="L",
                distances_file=distances_file,
                # cst_file=cst_file,
            ),
            # ProteinMPNNMetrics(
            #     chain="A", seed=1235, native_pdb=native_pdb, sampling_temp="0.1"
            # ),
        ]
        debug = True
        max_iteration = 50
        population_size = 50
        seed = 1235
        # eval_mutations_params = {
        #     "cst_file": cst_file,
        #     "min_energy_threshold": 0,
        #     "seed": seed,
        #     "params_folder": params_folder,
        #     "native_pdb": native_pdb,
        # }
        chains = ["A"]

        # Folder name and remove if exists
        folder_name = "mood_job"
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)

        with open(
            f"{data_path}/all_mutable_aa.json",
            "r",
        ) as f:
            design = json.load(f)
        with open(
            f"{data_path}/mutation_probs.json",
            "r",
        ) as f:
            mutations_probs = json.load(f)
        mutable_aa = {"A": design}
        mutations_probabilities = {"A": mutations_probs}
        fixed_positions = f"{data_path}/fixed_positions.json"

        self.moo = MultiObjectiveOptimization(
            optimizer=optimizer,
            metrics=metrics,
            debug=debug,
            max_iteration=max_iteration,
            native_pdb=native_pdb,
            chains=chains,
            data=data,
            mutable_aa=None,
            folder_name=folder_name,
            seed=seed,
            population_size=population_size,
            offset=1,
            # eval_mutations_params=eval_mutations_params,
            starting_sequences=starting_sequences,
            mutations_probabilities=mutations_probabilities,
            mutation_probability=True,
            # recombination_with_mutation=False,
            # fixed_positions=fixed_positions,
        )

    def test_run(self):
        self.moo.run()
        self.assertTrue(os.path.exists("mood_job/000"))


if __name__ == "__main__":
    unittest.main()
