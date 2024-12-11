import json

import numpy as np
import pandas as pd


def mutation_probabilities_calculation_proteinMPNN(
    chain, folder_name, native_pdb, seed, population_size, mutation_probabilities={}
):

    from mood.metrics.ProteinMPNN.protein_mpnn_run import mpnn_main

    # Run the ProteinMPNN

    out_folder = f"{folder_name}/input/mpnn_{chain}"
    mpnn_main(
        pdb_path=native_pdb,
        pdb_path_chains=chain,
        out_folder=out_folder,
        seed=seed,
        num_seq_per_target=population_size,
        sampling_temp="0.1",
        save_probs=True,
        suppress_print=True,
    )

    loaded_data = np.load(
        f"{out_folder}/probs/{native_pdb.split('/')[-1].split('.')[0]}.npz"
    )
    probs = loaded_data["probs"]

    # Define the alphabet
    alphabet = list("ACDEFGHIKLMNPQRSTVWYX")

    # Initialize arrays to store the sum of probabilities and sum of squared differences
    num_sequences = len(probs)
    num_positions = len(probs[0])
    num_amino_acids = len(alphabet)

    prob_sums = np.zeros((num_positions, num_amino_acids))
    # squared_diff_sums = np.zeros((num_positions, num_amino_acids))

    # Iterate through the sequences and accumulate the probabilities
    for seq in probs:
        for pos in range(num_positions):
            prob_sums[pos] += seq[pos]

    # Calculate the mean probabilities
    mean_probs = prob_sums / num_sequences

    # # Iterate through the sequences again to accumulate the squared differences
    # for seq in probs:
    #     for pos in range(num_positions):
    #         squared_diff_sums[pos] += (seq[pos] - mean_probs[pos]) ** 2

    # # Calculate the variance and then the standard deviation
    # variance = squared_diff_sums / num_sequences
    # std_deviation = np.sqrt(variance)

    # Get the dictionary of the probabilities

    for i in range(len(mean_probs)):
        mutation_probabilities[str(i)] = mean_probs[i].tolist()[:-1]

    with open(f"{out_folder}/mutation_probs.json", "w") as f:
        json.dump(mutation_probabilities, f)

    from Bio import SeqIO

    seq_proteinmpnn = []
    with open(
        f"{out_folder}/seqs/{native_pdb.split('/')[-1].split('.')[0]}.fa", "r"
    ) as f:
        seq_proteinmpnn = []
        for record in SeqIO.parse(f, "fasta"):
            seq_proteinmpnn.append(str(record.seq))

    return mutation_probabilities, seq_proteinmpnn
