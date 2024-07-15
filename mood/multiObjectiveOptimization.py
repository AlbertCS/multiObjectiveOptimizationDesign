""" This module contains the implementation of the Multi-Objective Optimization
    class. 
    This class will be in charge for the main logic of the program, 
    initializing the correct structures, distributing the work, and logging.
"""

import logging

import pandas as pd
from metrics import Metric
from optimizers import Optimizer


class moo:
    def __init__(
        self,
        optimizer: Optimizer = None,
        metrics: list[Metric] = None,
        debug: bool = False,
        max_iteration: int = 100,
        seed=12345,
    ) -> None:

        # Define the logger
        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%d-%b-%y %H:%M:%S",
                filename="mood.log",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%d-%b-%y %H:%M:%S",
                filename="mood.log",
            )

        # Initialize the optimizer and metrics
        self.optimizer = optimizer
        self.metrics = metrics
        self.max_iteration = max_iteration
        self.seed = seed

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

        for self.iteration in range(self.max_iteration):
            logging.info(f"***Starting iteration {self.iteration}***")
            if self.iteration == 0:
                logging.info("Initializing the first population")
                # Create the initial population
                self.optimizer.init_population()

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
            self.optimizer.eval_population(metric_df)

            # Create the child population
            logging.info("Creating the child population")
            self.optimizer.crossOver_mutate()
