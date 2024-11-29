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
        cpus = 20
        native_pdb = "tests/data/Felip9/FeLip9-PET-1.pdb"
        distances_file = "tests/data/distances.json"
        cst_file = "tests/data/Felip9/FeLip9-PET-1_CA.cst"
        starting_sequences = "tests/data/Felip9/ProtNPMM_native_small.fasta"

        # starting_sequences = "tests/data/Felip9/ProtNPMM_native_small.fasta"
        metrics = [
            RosettaMetrics(
                params_folder=params_folder,
                cpus=cpus,
                seed=1235,
                native_pdb=native_pdb,
                ligand_chain="L",
                distances_file=distances_file,
                cst_file=cst_file,
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
        with open(
            "tests/data/Felip9/all_mutable_aa.json",
            "r",
        ) as f:
            design = json.load(f)
        with open(
            "tests/data/Felip9/mutation_probs.json",
            "r",
        ) as f:
            mutations_probs = json.load(f)
        mutable_aa = {"A": design}
        mutations_probabilities = {"A": mutations_probs}
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
            # eval_mutations_params=eval_mutations_params,
            # starting_sequences=starting_sequences,
            mutations_probabilities=mutations_probabilities,
            recombination_with_mutation=False,
        )

    def test_run(self):
        self.moo.run()
        self.assertTrue(os.path.exists("mood_job/000"))


if __name__ == "__main__":
    unittest.main()
