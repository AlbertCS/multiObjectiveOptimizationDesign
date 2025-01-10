import json
import os
import subprocess

import pandas as pd

from mood.metrics import Metric

SCRIPT = """#! /bin/bash
singularity exec --bind %pdb_folder%:/pdb docker://proteinphysiologylab/frustratometer:latest /bin/bash -c "sh /script1.sh inicio configurational %license%"
singularity exec --bind %pdb_folder%:/pdb docker://proteinphysiologylab/frustratometer:latest /bin/bash -c "sh /script1.sh inicio singleresidue %license%"
"""


class FrustraRMetrics(Metric):
    def __init__(
        self,
        license_key,
    ):
        super().__init__()
        self.state = {
            "HighlyFrustratedIndex": "negative",
        }
        # This score is the summation of the highly frustrated residues (residues with positive energy)
        self._objectives = ["FrustrationTotal"]
        self.name = "frustraRMetrics"
        self.license_key = license_key

    @property
    def objectives(self):
        return self._objectives

    def compute(self, sequences, iteration, folder_name, chain=None):

        # Create df
        df = pd.DataFrame(sequences, columns=["Sequence"])

        sequences = df["Sequence"].tolist()

        # Create the folder for the frustration calculation
        output_folder = f"{folder_name}/{str(iteration).zfill(3)}/frustrar"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not os.path.exists(f"{output_folder}/pdb"):
            os.makedirs(f"{output_folder}/pdb")

        # Copy the pdbs from the relax folder to the frustration folder
        pdbs = []
        if os.path.exists(f"{folder_name}/{str(iteration).zfill(3)}/relax"):
            for file in os.listdir(f"{folder_name}/{str(iteration).zfill(3)}/relax"):
                if file.endswith(".pdb"):
                    name = (
                        file.split("_")[0]
                        + file.split("_")[2].replace("I", "")
                        + ".pdb"
                    )
                    pdbs.append(f"{folder_name}/{str(iteration).zfill(3)}/relax/{file}")
        pdbs.sort()
        names = []
        for pdb in pdbs:
            name_pdb = pdb.split("/")[-1]
            name = (
                name_pdb.split("_")[0]
                + name_pdb.split("_")[2].replace("I", "")
                + ".pdb"
            )
            names.append(name)
            os.system(f"cp {pdb} {f"{output_folder}/pdb/{name}"}")

        # Get the msa file
        # Specify the output file name
        sequences_file = f"{output_folder}/pdb/sequences.fasta"
        # Open the file in write mode
        with open(sequences_file, "w") as file:
            # Iterate over the sequences
            for sequence, name in zip(sequences, names):
                # Write each sequence to the file
                file.write(f">{name.replace(".pdb", "")}\n")
                file.write(sequence + "\n")

        # Run the frustration calculation
        script_value = SCRIPT
        script_value = script_value.replace("%pdb_folder%", f"{output_folder}/pdb")
        script_value = script_value.replace("%license%", self.license_key)
        with open(f"{output_folder}/frustra.sh", "w") as file:
            file.write(script_value)

        try:
            cmd = f"sh {output_folder}/frustra.sh"

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            stdout, stderr = proc.communicate()
            print("Output:", stdout.decode())
            print("Error:", stderr.decode())
            with open(f"{output_folder}/frustra.out", "w") as f:
                f.write(stdout.decode())
            with open(f"{output_folder}/frustra.err", "w") as f:
                f.write(stderr.decode())
        except Exception as e:
            raise Exception(f"An error occurred while running the Rosetta metrics: {e}")

        if not os.path.exists(
            f"{output_folder}/pdb/{names[0].replace(".pdb", "")}.done"
        ):
            raise ValueError("The FrustraR metrics did not run successfully")

        # Get the frustration values
        accu_frustration = {}
        accu_frst_index = []
        for name in names:
            with open(
                f"{output_folder}/pdb/{name.replace(".pdb", "")}.done/FrustrationData/{name}_configurational_5adens",
                "r",
            ) as file:
                configurational = pd.read_csv(file, sep=" ")
            with open(
                f"{output_folder}/pdb/{name.replace(".pdb", "")}.done/FrustrationData/{name}_singleresidue",
                "r",
            ) as file:
                frst_index = pd.read_csv(file, sep=" ")

            frustration = {}
            # Get if the position is frustrated or not
            for row in configurational.iterrows():
                if row[1].iloc[6] > row[1].iloc[7] and row[1].iloc[6] > row[1].iloc[8]:
                    # frustration.append("High")
                    frustration[row[1].iloc[1]] = row[1].iloc[6]
                elif (
                    row[1].iloc[7] > row[1].iloc[6] and row[1].iloc[7] > row[1].iloc[8]
                ):
                    # frustration.append("Neutral")
                    frustration[row[1].iloc[1]] = 0
                else:
                    # frustration.append("Minimal")
                    frustration[row[1].iloc[1]] = 0

            accu_frustration[name.replace(".pdb", "")] = frustration.copy()

            # Get the summation of the positive indexes
            index = 0
            for row in frst_index.iterrows():
                # HighlyFrustratedResidues
                if row[1].iloc[-1] > 0:
                    index += row[1].iloc[-1]
            accu_frst_index.append(index)

        # Create the dataframe
        df = pd.DataFrame(accu_frst_index, columns=["HighlyFrustratedIndex"])

        with open(f"{output_folder}/frust.json", "w") as f:
            json.dump(accu_frustration, f)
        accu_frustration
        # Add the sequence column
        df["Sequence"] = sequences

        return df
