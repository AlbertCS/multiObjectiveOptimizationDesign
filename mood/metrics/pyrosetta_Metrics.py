import io
import os
import subprocess

import pandas as pd
from pkg_resources import Requirement, resource_stream

from mood.metrics import Metric


class RosettaMetrics(Metric):
    def __init__(
        self,
        params_folder,
        cpus,
        seed,
        native_pdb,
        distances_file=None,
        cst_file=None,
        ligand_chain=None,
    ):
        super().__init__()
        self.state = {
            "Relax_Energy": "negative",
            "Hydrophobic_Score": "negative",
            "Salt_Bridges": "positive",
        }
        self._objectives = ["Relax_Energy"]
        self.name = "rosettaMetrics"
        self.params_folder = params_folder
        self.cpus = cpus
        self.seed = seed
        self.native_pdb = native_pdb
        self.distances_file = distances_file
        if self.distances_file is not None:
            self.state["distances"] = "negative"
        self.cst_file = cst_file
        self.ligand_chain = ligand_chain
        # If there is a ligand add the metrics related to the ligand
        if self.ligand_chain is not None:
            self._objectives.append("Interface_Score")
            self.state["Interface_Score"] = "negative"
            self.state["Apo_Score"] = "negative"

    @property
    def objectives(self):
        return self._objectives

    @objectives.setter
    def objectives(self, value):
        self._objectives = value

    def _copyScriptFile(
        self, output_folder, script_name, no_py=False, subfolder=None, hidden=True
    ):
        """
        Copy a script file from the MultiObjectiveOptimization package.

        Parameters
        ==========

        """
        # Get script
        base_path = "mood/metrics/scripts"
        if subfolder is not None:
            base_path = os.path.join(base_path, subfolder)

        script_path = os.path.join(base_path, script_name)
        with resource_stream(
            Requirement.parse("MultiObjectiveOptimization"), script_path
        ) as script_file:
            script_file = io.TextIOWrapper(script_file)

            # Adjust script name if no_py is True
            if no_py:
                script_name = script_name.replace(".py", "")

            # Build the output path
            if hidden:
                output_path = os.path.join(output_folder, f".{script_name}")
            else:
                output_path = os.path.join(output_folder, script_name)

            # Write the script to the output folder
            with open(output_path, "w") as sof:
                for line in script_file:
                    sof.write(line)

    def compute(self, sequences, iteration, folder_name):

        # Create df
        df = pd.DataFrame(sequences, columns=["Sequence"])

        sequences = df["Sequence"].tolist()

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

        # Copy the script to be run in mpi
        self._copyScriptFile(
            script_name="mpi_rosetta_metrics.py", output_folder=folder_name
        )

        # TODO may be a good idea to copy the params folder to the input folder of the mood

        try:
            cmd = f"mpirun -np {self.cpus} "
            cmd += f"python {folder_name}/.mpi_rosetta_metrics.py "
            cmd += f"--seed {self.seed} "
            cmd += f"--output_folder {output_folder} "
            cmd += f"--sequences_file {sequences_file} "
            cmd += f"--params_folder {self.params_folder} "
            cmd += f"--native_pdb {self.native_pdb} "
            cmd += f"--distances {self.distances_file} "
            cmd += f"--cst_file {self.cst_file} "
            cmd += f"--ligand_chain {self.ligand_chain}"

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            stdout, stderr = proc.communicate()
            print("Output:", stdout.decode())
            print("Error:", stderr.decode())
            with open(f"{output_folder}/rosetta.out", "w") as f:
                f.write(stdout.decode())
            with open(f"{output_folder}/rosetta.out", "w") as f:
                f.write(stdout.decode())
        except Exception as e:
            raise Exception(f"An error occurred while running the Rosetta metrics: {e}")

        # Delete the pdb files
        
        
        
        if not os.path.exists(f"{output_folder}/rosetta_scores.csv"):
            raise ValueError("The Rosetta metrics did not run successfully")
        else:
            df = pd.read_csv(f"{output_folder}/rosetta_scores.csv")
            # Add the sequence column
            df["Sequence"] = sequences

        return df
