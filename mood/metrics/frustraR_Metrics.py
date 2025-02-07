import json
import os
import subprocess

import pandas as pd

from mood.metrics import Metric

# singularity exec --bind %pdb_folder%:/pdb docker://proteinphysiologylab/frustratometer:latest /bin/bash -c "sh /script1.sh inicio configurational %license%"

SCRIPT = """#! /bin/bash
singularity exec --bind %pdb_folder%:/pdb %docker_image% /bin/bash -c "sh /script1.sh inicio singleresidue %license%"
"""


class FrustraRMetrics(Metric):
    def __init__(
        self,
        license_key,
        docker_image="docker://proteinphysiologylab/frustratometer:latest",
    ):
        super().__init__()
        self.state = {
            "Total_Frustration_Index": True,
            "Positions_Frustrated": False,
            "Totally_Max_Frst_Index": True,
        }
        # This score is the summation of the highly frustrated residues (residues with positive energy)
        self._objectives = ["Total_Frustration_Index"]
        self.name = "frustraRMetrics"
        self.license_key = license_key
        self.docker_image = docker_image

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
        # Clean frustration
        if iteration != 0 and iteration != max_iteration:
            frustrar_folder = f"{folder_name}/{str(iteration).zfill(3)}/frustrar"
            if os.path.exists(frustrar_folder):
                for root, dirs, files in os.walk(f"{frustrar_folder}/pdb"):
                    for file in files:
                        if file.endswith(".pdb"):
                            os.remove(os.path.join(root, file))

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
        if not os.path.exists(f"{output_folder}/results"):
            os.makedirs(f"{output_folder}/results")

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
                + name_pdb.split("_")[1].replace("I", "")
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
        script_value = script_value.replace("%docker_image%", self.docker_image)
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
        total_frustration_total = []
        for name in names:
            with open(
                f"{output_folder}/pdb/{name.replace(".pdb", "")}.done/FrustrationData/{name}_singleresidue",
                "r",
            ) as file:
                singleResidue = pd.read_csv(file, sep=" ")

            frustration_type = []
            total_pos_frustrated = 0
            total_frustration = 0
            total_max_frustrated = 0

            # Define the frustration type and if its highly frustrated make the summation
            for value in singleResidue["FrstIndex"]:
                if value > 0.55:
                    frustration_type.append("MIN")
                    total_frustration += value
                elif value < -1:
                    frustration_type.append("MAX")
                    total_pos_frustrated += 1
                    total_frustration += value
                    total_max_frustrated += value
                else:
                    frustration_type.append("NEU")
                    total_frustration += value

            aa = singleResidue["AA"]
            n_res = singleResidue["Res"]
            df = pd.DataFrame(
                {
                    "AA": aa,
                    "Res": n_res,
                    "FrstIndex": singleResidue["FrstIndex"],
                    "FrustrationType": frustration_type,
                }
            )

            df1 = pd.DataFrame({"Names": names, "Sequence": sequences})

            df1.to_csv(
                f"{output_folder}/results/equivalences.csv",
                index=False,
            )

            df.to_csv(
                f"{output_folder}/results/{name}_singleresidue.csv",
                index=False,
            )
            total_frustration_total.append(total_frustration)

        # Create the dataframe
        df = pd.DataFrame(
            {
                "Total_Frustration_Index": total_frustration_total,
                "Positions_Frustrated": total_pos_frustrated,
                "Totally_Max_Frst_Index": total_max_frustrated,
            }
        )

        # Add the sequence column
        df["Sequence"] = sequences

        return df
