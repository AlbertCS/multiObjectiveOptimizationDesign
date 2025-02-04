import json
import os

import numpy as np


def mutation_probabilities_calculation_proteinMPNN(
    chain,
    folder_name,
    native_pdb,
    seed,
    population_size,
    fixed_positions=None,
    generate_initial_seq=False,
    temp="0.1",
):

    from mood.metrics.ProteinMPNN.protein_mpnn_run import mpnn_main

    # Run the ProteinMPNN

    mutation_probabilities = {}
    out_folder = f"{folder_name}/input/mpnn_{chain}"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if fixed_positions:
        from mood.metrics.ProteinMPNN.parse_multiple_chains import (
            main_parse_multiple_chains,
        )

        path_for_parsed_pdbs = f"{out_folder}/parsed_pdbs.jsonl"
        path_for_fixed_positions = f"{out_folder}/fixed_positions.json"

        main_parse_multiple_chains(
            input_path=f"{folder_name}/input/scafold",
            output_path=path_for_parsed_pdbs,
        )

        if not os.path.exists(path_for_parsed_pdbs):
            raise ValueError(f"Fixed positions file not found: {path_for_parsed_pdbs}")

        from mood.metrics.ProteinMPNN.make_fixed_positions_dict import (
            main_make_fixed_positions,
        )

        main_make_fixed_positions(
            input_path=path_for_parsed_pdbs,
            output_path=path_for_fixed_positions,
            chain_list=chain,
            position_list=fixed_positions,
        )

        if not os.path.exists(path_for_fixed_positions):
            raise ValueError(
                f"Fixed positions file not found: {path_for_fixed_positions}"
            )

        # Get the conditional probabilities
        mpnn_main(
            jsonl_path=path_for_parsed_pdbs,
            pdb_path_chains=chain,
            fixed_positions_jsonl=path_for_fixed_positions,
            out_folder=out_folder,
            seed=seed,
            num_seq_per_target=1,
            sampling_temp=temp,
            suppress_print=True,
            conditional_probs_only=True,
        )

        if generate_initial_seq:
            # Calculate the 100 initial sequences
            mpnn_main(
                jsonl_path=path_for_parsed_pdbs,
                pdb_path_chains=chain,
                fixed_positions_jsonl=path_for_fixed_positions,
                out_folder=out_folder,
                seed=seed,
                num_seq_per_target=population_size - 1,
                sampling_temp=temp,
                suppress_print=True,
            )

    else:
        # Get the conditional probabilities
        print("Calculating probabilities")
        mpnn_main(
            pdb_path=native_pdb,
            pdb_path_chains=chain,
            out_folder=out_folder,
            seed=seed,
            num_seq_per_target=1,
            sampling_temp=temp,
            suppress_print=True,
            conditional_probs_only=True,
        )
        if generate_initial_seq:
            # Calculate the 100 initial sequences
            print("Generating initial seqs")
            mpnn_main(
                pdb_path=native_pdb,
                pdb_path_chains=chain,
                out_folder=out_folder,
                seed=seed,
                num_seq_per_target=population_size - 1,
                sampling_temp=temp,
                save_probs=True,
                suppress_print=True,
            )

    if not os.path.exists(
        f"{out_folder}/conditional_probs_only/{native_pdb.split('/')[-1].split('.')[0]}.npz"
    ):
        raise ValueError(
            f"Conditional probabilities file not found: {out_folder}/conditional_probs_only/{native_pdb.split('/')[-1].split('.')[0]}.npz"
        )

    loaded_data = np.load(
        f"{out_folder}/conditional_probs_only/{native_pdb.split('/')[-1].split('.')[0]}.npz"
    )

    # Load log-probabilities
    log_probs = loaded_data["log_p"]

    # Step 1: Exponentiate the log-probabilities to get probabilities
    probs = np.exp(log_probs)

    # Step 2: Normalize the probabilities (optional, as softmax already normalizes)
    probs /= probs.sum(axis=-1, keepdims=True)

    # Get the dictionary of the probabilities
    for i in range(len(probs[0])):
        mutation_probabilities[i] = probs[0][i].tolist()[:-1]

    with open(f"{out_folder}/mutation_probs.json", "w") as f:
        json.dump(mutation_probabilities, f)

    from Bio import SeqIO

    seq_proteinmpnn = []
    if not os.path.exists(
        f"{out_folder}/seqs/{native_pdb.split('/')[-1].split('.')[0]}.fa"
    ):
        raise ValueError(
            f"Sequences file not found: {out_folder}/seqs/{native_pdb.split('/')[-1].split('.')[0]}.fa"
        )
    with open(
        f"{out_folder}/seqs/{native_pdb.split('/')[-1].split('.')[0]}.fa", "r"
    ) as f:
        seq_proteinmpnn = []
        for record in SeqIO.parse(f, "fasta"):
            seq_proteinmpnn.append(str(record.seq))

    return mutation_probabilities, seq_proteinmpnn
