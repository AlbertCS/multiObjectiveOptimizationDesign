""" This module contains the implementation of the Multi-Objective Optimization
    class. 
    This class will be in charge for the main logic of the program, 
    initializing the correct structures, distributing the work, and logging.
"""

import argparse
import os
import pickle
import random
import shutil

import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio.Seq import Seq
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
        native_pdb=None,
        chains="A",
        data=None,
        mutation_rate=None,
        folder_name="mood_job",
        mutable_aa=None,
        population_size=100,
        offset=None,
    ) -> None:

        # Define the logger
        self.logger = Logger(debug).get_log()

        self.metrics = metrics
        self.max_iteration = max_iteration
        self.seed = seed
        random.seed(seed)
        self.native_pdb = native_pdb
        self.chains = chains
        if data is None:
            self.data = AlgorithmDataSingleton()
            self.data.chains = chains
        else:
            self.data = data
        self.mutation_rate = {}

        self.folder_name = folder_name

        self.folders = {}
        self.current_iteration = 0
        self.population_size = population_size
        # self.sequences = {chain: {Sequence1.sequence: Sequence1, Sequence2.sequence: Sequence2}}
        self.sequences = {chain: {} for chain in self.chains}
        self.sequences_file_name = "sequences.pkl"
        self.data_frame_file_name = "data_frame.pkl"

        if offset is None:
            self.mutable_aa = mutable_aa
        else:
            self.mutable_aa = {
                chain: {int(pos) - offset: aa for pos, aa in positions.items()}
                for chain, positions in mutable_aa.items()
            }
        if isinstance(chains, str):
            self.chains = [chains]
        elif isinstance(chains, list):
            self.chains = chains

        for chain in self.chains:
            if mutation_rate is None:
                self.mutation_rate[chain] = 1 / len(self.mutable_aa[chain].keys())

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
                    mutable_aa=self.mutable_aa,
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

        shutil.copy(self.native_pdb, self.folders["input"])

    def check_Iter_finished(self, iteration, sequences_pkl, dataframe_pkl):
        """
        Checks if the iteration is finished by checking if the sequences.pkl and data_frame.csv files are
        present and are not empty.

        Returns True if the iteration is finished.
        """
        # Check the sequences.pkl file
        if not os.path.exists(sequences_pkl):
            self.logger.error(f"File {sequences_pkl} not found")
            raise ValueError(f"File {sequences_pkl} not found")
        # Check if the sequences.pkl file is empty
        elif os.path.getsize(sequences_pkl) == 0:
            self.logger.error(f"File {sequences_pkl} is empty")
            raise ValueError(f"File {sequences_pkl} is empty")
        # Read the sequences.pkl file to check if has the correct number of sequences
        else:
            with open(sequences_pkl, "rb") as f:
                sequences = pickle.load(f)
                for chain in self.chains:
                    if len(sequences[chain]) != self.population_size:
                        self.logger.error(
                            f"The number of sequences in the {iteration} doesn't reach the population size"
                        )

        # Check the data_frame.csv file
        if not os.path.exists(dataframe_pkl):
            self.logger.error(f"File {dataframe_pkl} not found")
            raise ValueError(f"File {dataframe_pkl} not found")
        # Check if the data_frame.csv file is empty
        elif os.path.getsize(dataframe_pkl) == 0:
            self.logger.error(f"File {dataframe_pkl} is empty")
            raise ValueError(f"File {dataframe_pkl} is empty")
        else:
            with open(dataframe_pkl, "rb") as f:
                dataframe_iter = pickle.load(f)
                for chain in self.chains:
                    data_frame = pd.DataFrame(dataframe_iter[chain])
                    # Get the rows of the dataFrame and check if the number of sequences is correct
                    if data_frame.shape[0] != self.population_size:
                        self.logger.error(
                            f"The number of sequences in the dataframe {iteration} doesn't reach the population size"
                        )
        return True

    def check_previous_iterations(self):
        """
        Check if the results files exists for the previous iterations.
        For each iterations accumulate the sequences, and load the last df.
        Returns if the algorithm is finished, the total sequences, and the dataframe of the last iteration.
        Also updates the current_iteration attribute.
        """
        data_frame = {}
        # Check if the folder exists for the current iteration
        while os.path.exists(
            self.folder_name + "/" + str(self.current_iteration).zfill(3)
        ):
            self.logger.info(f"Iteration {self.current_iteration} data found")
            # Create the paths to the sequences.pkl and data_frame.csv
            sequences_pkl = (
                self.folder_name
                + "/"
                + str(self.current_iteration).zfill(3)
                + "/"
                + self.sequences_file_name
            )
            dataframe_pkl = (
                self.folder_name
                + "/"
                + str(self.current_iteration).zfill(3)
                + "/"
                + self.data_frame_file_name
            )
            # See if the iteration is finished
            finished = self.check_Iter_finished(
                self.current_iteration, sequences_pkl, dataframe_pkl
            )
            if finished:
                # Load the sequences
                # TODO we are loading the sequences in two points, here and in check_Iter_finished
                # TODO see if can be optimized to only one load.
                with open(sequences_pkl, "rb") as f:
                    sequences_iter = pickle.load(f)
                for chain in self.chains:
                    # self.sequences[chain] = {**self.sequences[chain], **sequences_iter[chain]}
                    self.sequences[chain].update(sequences_iter[chain])
                self.current_iteration += 1

                continue
        dataframe_pkl = (
            self.folder_name
            + "/"
            + str(self.current_iteration).zfill(3)
            + "/"
            + self.data_frame_file_name
        )
        if os.path.exists(dataframe_pkl):
            with open(dataframe_pkl, "rb") as f:
                sequences = pickle.load(f)
                data_frame = {}
                for chain in self.chains:
                    data_frame[chain] = pd.DataFrame(sequences[chain])
                    # Get the rows of the dataFrame and check if the number of sequences is correct

        # Send the sequences to the data class
        self.data.sequences = self.sequences

        # Return if finished, the total sequences, and the dataframe of the last iteration.
        if self.current_iteration < self.max_iteration:
            return False, data_frame
        else:
            return True, data_frame

    def save_iteration(self, current_iteration, sequences, dataframe):
        try:
            with open(
                self.folder_name
                + "/"
                + str(current_iteration).zfill(3)
                + "/"
                + self.sequences_file_name,
                "wb",
            ) as f:
                pickle.dump(sequences, f)
        except:
            self.logger.error(
                f"Error saving the sequences from iteration {current_iteration}"
            )
            raise ValueError(
                f"Error saving the sequences from iteration {current_iteration}"
            )
        try:
            with open(
                self.folder_name
                + "/"
                + str(current_iteration).zfill(3)
                + "/"
                + self.data_frame_file_name,
                "wb",
            ) as f:
                pickle.dump(dataframe, f)
        except:
            self.logger.error(
                f"Error saving the data_frame from iteration {current_iteration}"
            )
            raise ValueError(
                f"Error saving the data_frame from iteration {current_iteration}"
            )

    def select_parents(self, evaluated_sequences_df, percent_of_parents=0.25):
        # sort by rank, keep the 25% of the sequences as parents
        sequences_ranked = evaluated_sequences_df.sort_values(by="Ranks")
        top = sequences_ranked.head(int(len(sequences_ranked) * percent_of_parents))
        top_list = [seq for seq in top["Sequence"].tolist()]

        return top_list

    def setup_folders_iter(self, current_iteration):
        if not os.path.exists(self.folder_name + "/" + str(current_iteration).zfill(3)):
            os.mkdir(self.folder_name + "/" + str(current_iteration).zfill(3))

    def save_info(self, seq):
        info = {
            "native_sequence": seq,
            "chains": self.chains,
            "seed": self.seed,
            "mutation_rate": self.mutation_rate,
            "mutable_aa": self.mutable_aa,
            "population_size": self.population_size,
        }
        with open(
            self.folder_name + "/input/info.pkl",
            "wb",
        ) as f:
            pickle.dump(info, f)

    def load_info(self):
        with open(
            self.folder_name + "/input/info.pkl",
            "rb",
        ) as f:
            info = pickle.load(f)
        self.native_sequence = info["native_sequence"]
        self.chains = info["chains"]
        self.seed = info["seed"]
        self.mutation_rate = info["mutation_rate"]
        self.mutable_aa = info["mutable_aa"]
        self.population_size = info["population_size"]

    def run(self):

        # Run the optimization process
        self.logger.info("-------Starting the optimization process-------")
        self.logger.info("\tOptimizer: " + self.optimizer.__class__.__name__)
        self.logger.info(
            "\tMetrics: "
            + ", ".join([metric.__class__.__name__ for metric in self.metrics])
        )
        self.logger.info("\tMax Iteration: " + str(self.max_iteration))
        self.logger.info("\tPopulation Size: " + str(self.population_size))
        self.logger.info("\tSeed: " + str(self.seed))
        self.logger.info("------------------------------------------------")

        self.current_iteration = 0
        parents_sequences = {}

        # Get if the algorithm is finished, total_sequences, and the dataframe of the last iterations
        finished, evaluated_sequences_df = self.check_previous_iterations()
        if finished:
            self.logger.info(
                "Genetic algorithm has finished running. Increase the number of iterations to continue"
            )
            message = "Genetic algorithm has finished running. "
            message += "Increase the number of iterations to continue."
            print(message)
            return 0
        if self.current_iteration != 0:
            self.load_info()

        sequences_to_evaluate = {}

        for self.current_iteration in range(self.max_iteration):
            self.logger.info(f"***Starting iteration {self.current_iteration}***")

            if self.current_iteration == 0:
                self.logger.info("Setting up the folders")
                # Setup folders for the initial iteration
                self.setup_folders_initial()
                self.setup_folders_iter(self.current_iteration)
                self.logger.debug("Reading the input pdb")
                # Read the pdb file and get the sequences by chain
                seq_chains = self._get_seq_from_pdb(pdb_file=self.native_pdb)
                # Save natives sequences
                self.native_sequence = seq_chains
                self.optimizer.native = self.native_sequence
                # For each chain, initialize the population
                for chain in self.chains:
                    self.logger.info("Initializing the first population")
                    # Create the initial population
                    # TODO the sequences must be Seq objects
                    sequences_to_evaluate[chain] = self.optimizer.init_population(
                        chain, seq_chains[chain]
                    )

                # Save the info
                self.save_info(seq_chains)

            else:
                self.setup_folders_iter(self.current_iteration)
                # Get the sequences from the optimizer
                for chain in self.chains:
                    # Select the parent sequences
                    parents_sequences[chain] = self.select_parents(
                        evaluated_sequences_df[chain]
                    )

                    # Generate the child population
                    self.logger.info("Generating the child population")
                    sequences_to_evaluate[chain] = (
                        self.optimizer.generate_child_population(
                            parents_sequences[chain],
                            chain=chain,
                            current_iteration=self.current_iteration,
                        )
                    )

            # Calculate the metrics
            self.logger.info("Calculating the metrics")
            # TODO add information to the data_frame, iteration, mutations, etc
            evaluated_sequences_df = {}
            for chain in self.chains:
                parents_sequences = {}
                # Get the sequences as a string format
                sequences_to_evaluate_str = [
                    str(x) for x in sequences_to_evaluate[chain]
                ]
                metric_df = pd.DataFrame(
                    sequences_to_evaluate_str, columns=["Sequence"]
                )
                metric_df.set_index("Sequence", inplace=True)
                metric_states = {}
                metric_objectives = []
                metric_result = None
                for metric in self.metrics:
                    metric_result = metric.compute(
                        sequences=sequences_to_evaluate_str,
                        iteration=self.current_iteration,
                        folder_name=self.folder_name,
                    )
                    self.logger.info(f"Metric {metric.name} calculated")
                    metric_df = metric_df.merge(metric_result, on="Sequence")
                    metric_states[metric.name] = metric.state
                    metric_objectives.extend(metric.objectives)

                # Evaluate the population and rank the individuals
                self.logger.info("Evaluating and ranking the population")
                # Returns a dataframe with the sequences and the metrics, and a column with the rank
                evaluated_sequences_df[chain] = self.optimizer.eval_population(
                    df=metric_df,
                    metric_states=metric_states,
                    objectives=metric_objectives,
                )

            # Save the sequences and the data_frame
            self.save_iteration(
                self.current_iteration, sequences_to_evaluate, evaluated_sequences_df
            )
            self.current_iteration += 1

        self.logger.info("-------Finishing the optimization process-------")


if __name__ == "__main__":
    print("*Running the Multi-Objective Optimization module*")
    # metrics, population_size, mutation_rate, max_iterations, seed, debug = arg_parse()
    # TODO translate the metrics list to the respective objects
    # TODO translate the optimizer to the respective object
    data = AlgorithmDataSingleton()
    metrics = [Alphabet()]
    debug = True
    max_iterations = 10
    native_pdb = "/home/albertcs/GitHub/AlbertCS/multiObjectiveDesign/tests/data/NAGox-glucose-5.pdb"
    mood = MultiObjectiveOptimization(
        optimizer="genetic_algorithm",
        metrics=metrics,
        debug=debug,
        max_iteration=max_iterations,
        native_pdb=native_pdb,
        data=data,
    )
    mood.run()
    print("*Finished running the Multi-Objective Optimization module*")
