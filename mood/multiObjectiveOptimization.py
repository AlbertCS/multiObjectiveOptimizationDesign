""" This module contains the implementation of the Multi-Objective Optimization
    class. 
    This class will be in charge for the main logic of the program, 
    initializing the correct structures, distributing the work, and logging.
"""

import argparse
import logging
import os
import pickle
import random
import shutil

import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1

from mood.base.data import AlgorithmDataSingleton
from mood.base.log import Logger
from mood.metrics import Metric
from mood.metrics.alphabetic import Alphabet
from mood.optimizers import GeneticAlgorithm, Optimizer


def arg_parse():
    parser = argparse.ArgumentParser(description="Train the models")

    parser.add_argument(
        "-m",
        "--metrics",
        required=True,
        default=["total_energy"],
        type=list,
        help="The metrics to evaluate",
    )
    parser.add_argument(
        "-p",
        "--population_size",
        required=False,
        default=100,
        type=int,
        help="The total population size of the genetic algorithm",
    )
    parser.add_argument(
        "-m",
        "--mutation_rate",
        required=False,
        default=50,
        type=float,
        help="The number of threads to search for the hyperparameter space",
    )
    parser.add_argument(
        "-i",
        "--max_iterations",
        required=False,
        default=100,
        type=int,
        help="The maximum number of iterations to run the optimization",
    )
    parser.add_argument(
        "-s",
        "--seed",
        required=False,
        default=12345,
        type=int,
        help="The seed for the random number generator",
    )
    parser.add_argument(
        "-d",
        "--debug",
        required=False,
        default=False,
        type=bool,
        help="The flag to enable the debug mode",
    )

    args = parser.parse_args()

    return [
        args.metrics,
        args.population_size,
        args.mutation_rate,
        args.max_iterations,
        args.seed,
        args.debug,
    ]


class MultiObjectiveOptimization:
    def __init__(
        self,
        optimizer: str = None,
        metrics: list[Metric] = None,
        debug: bool = False,
        max_iteration: int = 100,
        seed=12345,
        pdb=None,
        chains="A",
        data=None,
        mutation_rate=None,
        mutable_positions=None,
        folder_name="mood_job",
        mutable_aa=None,
        population_size=100,
    ) -> None:

        # Define the logger
        self.logger = Logger(debug).get_log()

        self.metrics = metrics
        self.max_iteration = max_iteration
        random.seed(seed)
        self.pdb = pdb
        self.chains = chains
        self.data = data
        self.mutation_rate = {}
        self.mutable_positions = mutable_positions
        self.folder_name = folder_name
        self.mutable_aa = mutable_aa
        self.folders = {}
        self.current_iteration = 0

        if isinstance(chains, str):
            self.chains = [chains]
        elif isinstance(chains, list):
            self.chains = chains

        for chain in self.chains:
            if mutation_rate is None:
                self.mutation_rate[chain] = 1 / len(self.mutable_positions[chain])

        # Initialize the optimizer and metrics
        available_optimizers = ["genetic_algorithm"]
        if optimizer not in available_optimizers:
            raise ValueError(
                f"Optimizer {optimizer} not available. Choose from {available_optimizers}"
            )
        else:
            if optimizer == "genetic_algorithm":
                self.optimizer = GeneticAlgorithm(
                    population_size=population_size,
                    seed=seed,
                    debug=debug,
                    data=data,
                    mutable_positions=mutable_positions,
                    mutable_aa=mutable_aa,
                )

    def _get_seq_from_pdb(self, structure_id="initial_structure", pdb_file=None):

        parser = PDBParser()
        structure = parser.get_structure(structure_id, pdb_file)
        chains = {
            chain.id: seq1("".join(residue.resname for residue in chain))
            for chain in structure.get_chains()
        }
        return chains

    def setup_folders_initial(self):
        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)
        self.folders["job"] = self.folder_name

        if not os.path.exists(self.folder_name + "/input"):
            os.mkdir(self.folder_name + "/input")
        self.folders["input"] = self.folder_name + "/input"

        shutil.copy(self.pdb, self.folders["input"])

    def setup_folders(self, current_iteration):

        pass

    def check_Iter_finished(self, iteration, sequences_pkl, dataframe_pkl):
        """
        Checks if the iteration is finished by checking if the sequences.pkl and data_frame.csv files are
        present and are not empty.

        Returns True if the iteration is finished.
        """
        # Check the sequences.pkl file
        if not os.path.exists(sequences_pkl):
            logging.error(f"File {sequences_pkl} not found")
            raise ValueError(f"File {sequences_pkl} not found")
        # Check if the sequences.pkl file is empty
        elif os.path.getsize(sequences_pkl) == 0:
            logging.error(f"File {sequences_pkl} is empty")
            raise ValueError(f"File {sequences_pkl} is empty")
        # Read the sequences.pkl file to check if has the correct number of sequences
        else:
            with open(sequences_pkl, "rb") as f:
                sequences = pickle.load(f)
                for chain in self.chains:
                    if len(sequences[chain]) != self.population_size:
                        logging.error(
                            f"The number of sequences in the {iteration} doesn't reach the population size"
                        )

        # Check the data_frame.csv file
        if not os.path.exists(dataframe_pkl):
            logging.error(f"File {dataframe_pkl} not found")
            raise ValueError(f"File {dataframe_pkl} not found")
        # Check if the data_frame.csv file is empty
        elif os.path.getsize(dataframe_pkl) == 0:
            logging.error(f"File {dataframe_pkl} is empty")
            raise ValueError(f"File {dataframe_pkl} is empty")
        else:
            # TODO if that works
            with open(dataframe_pkl, "rb") as f:
                sequences = pickle.load(f)
                for chain in self.chains:
                    data_frame = pd.DataFrame(sequences[chain])
                    # Get the rows of the dataFrame and check if the number of sequences is correct
                    if data_frame.shape[0] != self.population_size:
                        logging.error(
                            f"The number of sequences in the {iteration} doesn't reach the population size"
                        )
        return True

    def check_previous_iterations(self):
        sequences = {}
        while os.path.exists(
            self.folder_name + "/" + str(self.current_iteration).zfill(3)
        ):
            logging.info(f"Iteration {self.current_iteration} data found")
            # Create the paths to the sequences.pkl and data_frame.csv
            sequences_pkl = (
                self.folder_name
                + "/"
                + str(self.current_iteration).zfill(3)
                + "/sequences.pkl"
            )
            dataframe_pkl = (
                self.folder_name
                + "/"
                + str(self.current_iteration).zfill(3)
                + "/data_frame.pkl"
            )
            # See if the iteration is finished
            finished = self.check_Iter_finished(
                self.current_iteration, sequences_pkl, dataframe_pkl
            )
            if finished:
                # Load the sequences

                with open(sequences_pkl, "rb") as f:
                    sequences_iter = f.readlines()

                sequences = {**sequences, **sequences_iter}
                self.current_iteration += 1

                continue

        return sequences

    def save_iteration(self, parents_sequences):
        try:
            with open(
                self.folder_name
                + "/"
                + str(self.current_iteration).zfill(3)
                + "/sequences.pkl",
                "wb",
            ) as f:
                f.write(parents_sequences)
        except:
            logging.error(
                f"Error saving the sequences and data_frame from iteration {self.current_iteration}"
            )
            raise ValueError(
                f"Error saving the sequences and data_frame from iteration {self.current_iteration}"
            )

    def run(self):

        # Run the optimization process
        logging.info("-------Starting the optimization process-------")
        logging.info("\tOptimizer: " + self.optimizer.__class__.__name__)
        logging.info(
            "\tMetrics: "
            + ", ".join([metric.__class__.__name__ for metric in self.metrics])
        )
        logging.info("\tMax Iteration: " + str(self.max_iteration))
        logging.info("\tSeed: " + str(self.seed))
        logging.info("------------------------------------------------")

        self.current_iteration = 0
        parents_sequences = None

        # TODO if we start from a previous execution
        # Get the sequences, the chains, iterations
        # _, _ = self.check_previous_iterations()
        sequences_to_evaluate = {}
        self.chains = []

        if self.current_iteration == self.max_iteration + 1:
            message = "Genetic algorithm has finished running. "
            message += "Increase the number of iterations to continue."
            print(message)
            return
        # TODO save native sequence
        # TODO save the sequences obtained from the ga to the data.data_frame
        for self.current_iteration in range(self.max_iteration):
            logging.info(f"***Starting iteration {self.current_iteration}***")

            if self.current_iteration == 0:
                logging.info("Setting up the folders")
                self.setup_folders(self.current_iteration)
                logging.debug("Reading the input pdb")
                # Read the pdb file and get the sequences by chain
                seq_chains = self._get_seq_from_pdb(pdb_file=self.pdb)
                # For each chain, initialize the population
                for chain in self.chains:
                    logging.info("Initializing the first population")
                    # Create the initial population
                    # TODO the sequences must be Seq objects
                    sequences_to_evaluate[chain] = self.optimizer.init_population(
                        seq_chains[chain]
                    )

            else:
                # Get the sequences from the optimizer
                if parents_sequences is None:
                    raise ValueError("No parent sequences found")
                for chain in self.chains:
                    logging.info("Generating the child population")
                    sequences_to_evaluate[chain] = (
                        self.optimizer.generate_child_population(
                            parents_sequences[chain],
                            self.mutable_positions[chain],
                        )
                    )

            # Calculate the metrics
            logging.info("Calculating the metrics")
            # TODO add information to the data_frame, iteration, mutations, etc
            for chain in self.chains:
                parents_sequences = {}
                sequences_eval_df = pd.DataFrame()
                metric_df = pd.DataFrame(
                    sequences_to_evaluate[chain], columns=["Sequence"]
                )
                metric_df.set_index("Sequence", inplace=True)
                for metric in self.metrics:
                    metric_result = metric.calculate(sequences_to_evaluate)
                    metric_df = metric_df.merge(metric_result, on="Sequence")

                # Evaluate the population and rank the individuals
                logging.info("Evaluating and ranking the population")
                # Returns a dataframe with the sequences and the metrics, and a column with the rank
                sequences_eval_df[chain] = self.optimizer.eval_population(metric_df)
                # TODO select the parent population from the sequences_eval_df

                # TODO save the parents_sequences[chain] and metric_df as a csv:chain

            # Save the sequences and the data_frame
            # self.save_iteration(parents_sequences)

        # TODO check if the last evaluation is necessary
        logging.info("-------Finishing the optimization process-------")


if __name__ == "__main__":
    print("*Running the Multi-Objective Optimization module*")
    # metrics, population_size, mutation_rate, max_iterations, seed, debug = arg_parse()
    # TODO translate the metrics list to the respective objects
    # TODO translate the optimizer to the respective object
    data = AlgorithmDataSingleton()
    metrics = [Alphabet()]
    debug = True
    max_iterations = 10
    pdb = "/home/albertcs/GitHub/AlbertCS/multiObjectiveDesign/tests/data/NAGox-glucose-5.pdb"
    mood = MultiObjectiveOptimization(
        optimizer="genetic_algorithm",
        metrics=metrics,
        debug=debug,
        max_iteration=max_iterations,
        pdb=pdb,
        data=data,
    )
    mood.run()
    print("*Finished running the Multi-Objective Optimization module*")
