import logging

from optimizers.optimizers import Optimizer


class GeneticAlgorithm(Optimizer):
    def __init__(
        self,
        population_size=100,
        mutation_rate=0.005,
        seed=12345,
        debug=False,
        data=None,
        mutation_seq_percent=0.5,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.debug = debug
        self.data = data
        self.mutation_seq_percent = mutation_seq_percent

    def init_population(self, sequences_initial):
        logging.info("Initializing the population")
        try:
            # Getting the number of missing sequences
            n_missing = self.population_size - len(sequences_initial)
            # Calculating the index of the next sequence to generate
            index = self.population_size - n_missing + 1
            # TODO finish commenting and generating the missing sequences
            if n_missing == 0:
                logging.info("Population already at the desired size")
            else:
                logging.info(f"Adding {n_missing} random sequences to the population")
                logging.info(
                    f"Populating {self.mutation_seq_percent * 100}% of the {self.population_size} total population"
                )
                if index > self.population_size * self.mutation_seq_percent:
                    logging.info(
                        f"Population already at {self.mutation_seq_percent * 100}%, no mutations will occur"
                    )
                while index <= self.population_size * self.mutation_seq_percent:
                    logging.debug(f"Adding sequence {index} to the population")
                    if len(sequences_initial) > 1:
                        sequence_to_start_from = sequences_initial[
                            self.rng.randrange(1, len(sequences_initial))
                        ]
                    else:
                        sequence_to_start_from = sequences_initial[0]
                    mutated_sequence = self.generate_mutation_sequence(
                        sequence_to_start_from
                    )
                    self.data.add_sequence(mutated_sequence)
        except Exception as e:
            logging.error(f"Error initializing the population: {e}")

    def generate_mutation_sequence(self):
        logging.debug("Generating a mutation sequence")

    def generate_crossOver_sequence(self):
        logging.debug("Generating a crossOver sequence")

    def get_sequences(self):
        print("Getting the sequences")

    def eval_population(self):
        print("Evaluating the population")

    def crossOver_mutate(self):
        print("Crossing over and mutating the population")
