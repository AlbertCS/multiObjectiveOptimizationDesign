""" This module contains the implementation of the Multi-Objective Optimization
    class. 
    This class will be in charge for the main logic of the program, 
    initializing the correct structures, distributing the work, and logging.
"""

import argparse
import logging
import random

import pandas as pd
from base.data import AlgorithmDataSingleton
from metrics import Metric
from metrics.alphabetic import Alphabet
from optimizers import GeneticAlgorithm, Optimizer


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


class moo:
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
        mutatable_positions=None,
    ) -> None:

        # Define the logger
        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%d-%m-%y %H:%M:%S",
                filemode="w",
                filename="mood.log",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%d-%m-%y %H:%M:%S",
                filemode="w",
                filename="mood.log",
            )

        # Initialize the optimizer and metrics
        available_optimizers = ["genetic_algorithm"]
        if optimizer not in available_optimizers:
            raise ValueError(
                f"Optimizer {optimizer} not available. Choose from {available_optimizers}"
            )
        else:
            
        self.optimizer = Optimizer()
        self.metrics = metrics

        self.max_iteration = max_iteration
        random.seed(seed)
        self.pdb = pdb
        self.chains = chains
        self.data = data
        self.mutation_rate = {}

        if isinstance(chains, str):
            self.chains = [chains]
        elif isinstance(chains, list):
            self.chains = chains

        for chain in self.chains:
            if mutation_rate is None:
                self.mutation_rate[chain] = 1 / len(self.mutatable_positions[chain])
                self.optimizer.mutation_rate = mutation

    def _get_seq_from_pdb(self, structure_id="initial_structure", pdb_file=None):
        from Bio.PDB.PDBParser import PDBParser
        from Bio.SeqUtils import seq1

        parser = PDBParser()
        structure = parser.get_structure(structure_id, pdb_file)
        chains = {
            chain.id: seq1("".join(residue.resname for residue in chain))
            for chain in structure.get_chains()
        }
        return chains

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

        if self.current_iteration == self.max_iteration + 1:
            message = "Genetic algorithm has finished running. "
            message += "Increase the number of iterations to continue."
            print(message)
            return

        for self.current_iteration in range(self.max_iteration):
            logging.info(f"***Starting iteration {self.current_iteration}***")
            if self.current_iteration == 0:
                logging.debug("Reading the input pdb")
                # Read the pdb file and get the sequences by chain
                seq_chains = self._get_seq_from_pdb(pdb_file=self.pdb)
                # For each chain, initialize the population
                for chain in self.chains:
                    logging.info("Initializing the first population")
                    # Create the initial population
                    self.optimizer.init_population(seq_chains[chain])

            else:
                # Get the sequences from the optimizer
                sequences = self.optimizer.get_sequences()

            # Calculate the metrics
            logging.info("Calculating the metrics")
            metric_df = pd.DataFrame(sequences, columns=["Sequence"])
            metric_df.set_index("Sequence", inplace=True)
            for metric in self.metrics:
                metric_result = metric.calculate(sequences)
                metric_df = metric_df.merge(metric_result, on="Sequence")

            # Evaluate the population and rank the individuals
            logging.info("Evaluating and ranking the population")
            # self.optimizer.eval_population(metric_df)
            self.optimizer.eval_population()

            # Create the child population
            logging.info("Creating the child population")
            self.optimizer.crossOver_mutate()

        logging.info("-------Finishing the optimization process-------")


if __name__ == "__main__":
    print("*Running the Multi-Objective Optimization module*")
    # metrics, population_size, mutation_rate, max_iterations, seed, debug = arg_parse()
    # TODO translate the metrics list to the respective objects
    # TODO translate the optimizer to the respective object
    data = AlgorithmDataSingleton()
    ga = GeneticAlgorithm(data=data)
    metrics = [Alphabet("folder")]
    debug = True
    max_iterations = 10
    pdb = "/home/albertcs/GitHub/AlbertCS/multiObjectiveDesign/tests/data/NAGox-glucose-5.pdb"
    mood = moo(
        optimizer=ga,
        metrics=metrics,
        debug=debug,
        max_iteration=max_iterations,
        pdb=pdb,
        data=data,
    )
    mood.run()
    print("*Finished running the Multi-Objective Optimization module*")
