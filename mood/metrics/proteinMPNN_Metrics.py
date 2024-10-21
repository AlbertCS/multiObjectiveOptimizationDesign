import io
import os

import numpy as np
import pandas as pd
from pkg_resources import Requirement, resource_stream

from mood.metrics import Metric
from mood.metrics.ProteinMPNN.protein_mpnn_run import mpnn_main


class ProteinMPNNMetrics(Metric):
    def __init__(
        self,
        chain,
        seed,
        native_pdb,
        num_seq_per_target=5,
        sampling_temp="0.1",
        score_only=True,
        batch_size=1,
    ):
        super().__init__()
        self.state = {
            "ScoreMPNN": "negative",
        }
        # The lower the score, the better the fit of the designed sequence to the protein structure.
        self._objectives = ["ScoreMPNN"]
        self.name = "proteinMPNNMetrics"
        self.seed = seed
        self.native_pdb = native_pdb
        self.chain = chain
        self.num_seq_per_target = num_seq_per_target
        self.sampling_temp = sampling_temp
        self.score_only = score_only
        self.batch_size = batch_size

    @property
    def objectives(self):
        return self._objectives

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

        output_folder = f"{folder_name}/{str(iteration).zfill(3)}/mpnn"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Specify the output file name
        sequences_file = f"{output_folder}/sequences.fasta"
        # Open the file in write mode
        with open(sequences_file, "w") as file:
            # Iterate over the sequences
            for i, sequence in enumerate(sequences):
                # Write each sequence to the file
                file.write(f">s{i}\n")
                file.write(sequence + "\n")

        # Run the ProteinMPNN
        mpnn_main(
            path_to_fasta=sequences_file,
            pdb_path=self.native_pdb,
            pdb_path_chains=self.chain,
            out_folder=output_folder,
            seed=self.seed,
            num_seq_per_target=self.num_seq_per_target,
            sampling_temp=self.sampling_temp,
            score_only=self.score_only,
            batch_size=self.batch_size,
            suppress_print=True,
        )

        if not os.path.exists(f"{output_folder}/score_only"):
            raise ValueError("The ProteinMPNN metrics did not run successfully")
        else:
            mean_scores = []
            for file in os.listdir(f"{output_folder}/score_only"):
                if file.endswith("_pdb.npz"):
                    continue
                loaded_data = np.load(f"{output_folder}/score_only/{file}")
                mean_scores.append(loaded_data["score"].mean())

            # Create the dataframe
            df = pd.DataFrame(mean_scores, columns=["ScoreMPNN"])
            # Add the sequence column
            df["Sequence"] = sequences

        return df
