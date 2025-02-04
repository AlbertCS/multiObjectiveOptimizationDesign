import json
import os

import numpy as np
import pandas as pd

from mood.metrics import Metric
from mood.metrics.ProteinMPNN.protein_mpnn_run import mpnn_main


class ProteinMPNNMetrics(Metric):
    def __init__(
        self,
        seed,
        native_pdb,
        num_seq_per_target=1,
        sampling_temp="0.1",
        score_only=True,
        batch_size=1,
        fixed_positions=None,
    ):
        super().__init__()
        self.state = {
            "ScoreMPNN": False,
        }
        # The lower the score, the better the fit of the designed sequence to the protein structure.
        self._objectives = ["ScoreMPNN"]
        self.name = "proteinMPNNMetrics"
        self.seed = seed
        self.native_pdb = native_pdb
        self.num_seq_per_target = num_seq_per_target
        self.sampling_temp = sampling_temp
        self.score_only = score_only
        self.batch_size = batch_size
        self.fixed_positions = {}
        if fixed_positions is not None:
            with open(fixed_positions, "r") as f:
                self.fixed_positions = json.load(f)

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
        mpnn_folder = f"{folder_name}/{str(iteration).zfill(3)}/mpnn"
        if os.path.exists(mpnn_folder):
            for file in os.listdir(mpnn_folder):
                if file.endswith(".fasta"):
                    os.remove(f"{mpnn_folder}/{file}")

    def compute(self, sequences, iteration, folder_name, chain):

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

        # Not needed for the evaluation only when creating the sequences
        # if self.fixed_positions:
        #     from mood.metrics.ProteinMPNN.parse_multiple_chains import (
        #         main_parse_multiple_chains,
        #     )

        #     main_parse_multiple_chains(
        #         input_path=f"{folder_name}/input",
        #         output_path=f"{output_folder}/parsed_pdbs.jsonl",
        #     )

        #     from mood.metrics.ProteinMPNN.make_fixed_positions_dict import (
        #         main_make_fixed_positions,
        #     )

        #     main_make_fixed_positions(
        #         input_path=f"{output_folder}/parsed_pdbs.jsonl",
        #         output_path=f"{output_folder}/fixed_positions.json",
        #         chain_list=chain,
        #         position_list=self.fixed_positions[chain],
        #     )

        # Run the ProteinMPNN
        mpnn_main(
            path_to_fasta=sequences_file,
            pdb_path=self.native_pdb,
            pdb_path_chains=chain,
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
