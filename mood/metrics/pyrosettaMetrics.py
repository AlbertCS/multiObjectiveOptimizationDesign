import io
import os
import subprocess

import pandas as pd
from pkg_resources import Requirement, resource_listdir, resource_stream

from mood.metrics import Metric


class RosettaMetrics(Metric):
    def __init__(self, params_folder, cpus, seed, native_pdb, distances_file):
        super().__init__()
        self.state = {
            "RelaxEnergy": "negative",
            "InterfaceScore": "negative",
            "Apo Score": "negative",
            "HydrophobicScore": "negative",
            "SaltBridges": "positive",
            "distances": "negative",
        }
        self.name = "rosettaMetrics"
        self.params_folder = params_folder
        self.cpus = cpus
        self.seed = seed
        self.native_pdb = native_pdb
        self.distances_file = distances_file

    def _copyScriptFile(
        self, output_folder, script_name, no_py=False, subfolder=None, hidden=True
    ):
        """
        Copy a script file from the MultiObjectiveOptimization package.

        Parameters
        ==========

        """
        # Get script
        path = "mood/metrics/scripts"
        if subfolder != None:
            path = path + "/" + subfolder

        script_file = resource_stream(
            Requirement.parse("MultiObjectiveOptimization"), path + "/" + script_name
        )
        script_file = io.TextIOWrapper(script_file)

        # Write control script to output folder
        if no_py == True:
            script_name = script_name.replace(".py", "")

        if hidden:
            output_path = output_folder + "/._" + script_name
        else:
            output_path = output_folder + "/" + script_name

        with open(output_path, "w") as sof:
            for l in script_file:
                sof.write(l)

    def compute(self, sequences):

        # Create df
        df = pd.DataFrame(sequences, columns=["Sequence"])

        sequences = df["Sequence"].tolist()

        # Specify the output file name
        output_file = "sequences.txt"
        # Open the file in write mode
        with open(output_file, "w") as file:
            # Iterate over the sequences
            for sequence in sequences:
                # Write each sequence to the file
                file.write(sequence + "\n")

        # Copy the script to be run in mpi
        self._copyScriptFile(script_name="mpi_rosetta_metrics.py", output_folder=".")

        try:
            proc = subprocess.Popen(
                [
                    f"mpirun -np {self.cpus}",
                    "python ",
                    "mpi_rosetta_metrics.py",
                    f"--seed {self.seed}",
                    f"--params_folder {self.params_folder}",
                    f"--native_pdb {self.native_pdb}",
                    f"--distances {self.distances_file}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate()
            print("Output:", stdout.decode())
            print("Error:", stderr.decode())
        except Exception as e:
            raise Exception(f"An error occurred while running the Rosetta metrics: {e}")

        if not os.path.exists("sequences_energy.csv"):
            raise ValueError("The Rosetta metrics did not run successfully")
        else:
            df = pd.read_csv("sequences_energy.csv")
            # Add the sequence column
            df["Sequence"] = sequences

        return df
