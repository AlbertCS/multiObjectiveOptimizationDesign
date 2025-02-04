import io
import os

import pandas as pd
from pkg_resources import Requirement, resource_stream

from mood.metrics import Metric
from mood.metrics.scripts.thread_rosetta_metrics import ProcessRelax


class RosettaMetrics(Metric):
    def __init__(
        self,
        cpus,
        seed,
        native_pdb,
        params_folder=None,
        distances_file=None,
        cst_file=None,
        ligand_chain=None,
    ):
        super().__init__()
        # State if the state is positive then is true
        self.state = {
            "Relax_Energy": False,
            "Hydrophobic_Score": False,
            "Salt_Bridges": True,
        }
        self._objectives = ["Relax_Energy"]
        self.name = "rosettaMetrics"
        self.params_folder = params_folder
        self.cpus = cpus
        self.seed = seed
        self.native_pdb = native_pdb
        self.distances_file = distances_file
        if self.distances_file is not None:
            self.state["distances"] = False
        self.cst_file = cst_file
        self.ligand_chain = ligand_chain
        # If there is a ligand add the metrics related to the ligand
        if self.ligand_chain is not None:
            self._objectives.append("Interface_Score")
            self.state["Interface_Score"] = False
            self.state["Apo_Score"] = False

    @property
    def objectives(self):
        return self._objectives

    @objectives.setter
    def objectives(self, value):
        if not isinstance(value, list):
            raise ValueError("Objectives must be a list")
        if not all(isinstance(item, str) for item in value):
            raise ValueError("All objectives must be strings")
        self._objectives = value

    def clean(self, folder_name, iteration, max_iteration):
        if iteration != max_iteration:
            relax_folder = f"{folder_name}/{str(iteration).zfill(3)}/relax"
            if os.path.exists(relax_folder):
                for file in os.listdir(relax_folder):
                    if file.endswith(".pdb") or file.endswith(".pdb.gz"):
                        os.remove(f"{relax_folder}/{file}")

    def compute(self, sequences, iteration, folder_name, chain):

        # Create df
        df = pd.DataFrame(sequences, columns=["Sequence"])

        sequences = df["Sequence"].tolist()

        if sequences == []:
            raise ValueError("No sequences to evaluate")

        output_folder = f"{folder_name}/{str(iteration).zfill(3)}/relax"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Specify the output file name
        sequences_file = f"{output_folder}/sequences.txt"
        # Open the file in write mode
        with open(sequences_file, "w") as file:
            # Iterate over the sequences
            for sequence in sequences:
                # Write each sequence to the file
                file.write(sequence + "\n")

        # TODO may be a good idea to copy the params folder to the input folder of the mood
        try:
            if self.params_folder is None:
                patches = []
                params = []
            elif not os.path.exists(self.params_folder):
                print(f"Warning: Directory {self.params_folder} does not exist")
                patches = []
                params = []
            else:
                patches = [
                    self.params_folder + "/" + x
                    for x in os.listdir(self.params_folder)
                    if x.endswith(".txt")
                ]
                params = [
                    self.params_folder + "/" + x
                    for x in os.listdir(self.params_folder)
                    if x.endswith(".params")
                ]

            # "-relax:range:cycles 1000"
            options = f"-relax:default_repeats 1 -relax:range:cycles 5 -constant_seed true -jran {self.seed}"
            if params != []:
                params = " ".join(params)
                options += f" -extra_res_fa {params}"
            if patches != []:
                patches = " ".join(patches)
                options += f" -extra_patch_fa {patches}"

            process_relax = ProcessRelax(options)
            process_relax.main(
                output_folder=output_folder,
                sequences_file=sequences_file,
                native_pdb=self.native_pdb,
                distance_file=self.distances_file,
                cst_file=self.cst_file,
                ligand_chain=self.ligand_chain,
                n_processes=self.cpus,
            )

        except Exception as e:
            raise Exception(f"An error occurred while running the Rosetta metrics: {e}")

        if not os.path.exists(f"{output_folder}/rosetta_scores.csv"):
            raise ValueError("The Rosetta metrics did not run successfully")
        else:
            df = pd.read_csv(f"{output_folder}/rosetta_scores.csv")
            # Add the sequence column
            df["Sequence"] = sequences

        return df
