import logging
import random

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
        mutable_positions=[],
        mutable_aa={},
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        # self.seed = seed
        self.debug = debug
        self.data = data
        self.mutation_seq_percent = mutation_seq_percent
        random.seed(seed)
        self.mutable_positions = mutable_positions
        self.mutable_aa = mutable_aa

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
                if index > self.population_size * self.mutation_seq_percent:
                    logging.info(
                        f"Population already at {self.mutation_seq_percent * 100}%, no mutations will occur"
                    )
                else:
                    logging.info(
                        f"Populating {self.mutation_seq_percent * 100}% of the {self.population_size} total population"
                    )
                # Adding sequences by mutation til the desired percentage is reached
                while index <= self.population_size * self.mutation_seq_percent:
                    logging.debug(f"Adding sequence {index} to the population")
                    # Check if the input sequences of the algorithm are more than one, then select one random.
                    if len(sequences_initial) > 1:
                        sequence_to_start_from = sequences_initial[
                            self.rng.randrange(1, len(sequences_initial))
                        ]
                    else:
                        sequence_to_start_from = sequences_initial[0]
                    mutated_sequence = self.generate_mutation_sequence(
                        sequence_to_start_from
                    )
                    # Add the new sequence to the data object
                    self.data.add_sequence(mutated_sequence)
                    index += 1
                    number_of_sequences = len(self.data.sequences)
                    logging.debug(
                        f"Population size: {number_of_sequences} == {index} :index"
                    )

                # Adding sequences by crossOver til the desired population size is reached
                number_of_sequences = len(self.data.sequences)
                while number_of_sequences < self.population_size:
                    logging.debug(
                        f"Adding sequence {number_of_sequences} to the population by CrossOver"
                    )
                    # Get two random sequences to crossOver
                    sequence_keys = list(self.data.sequences.keys())
                    sequence1_id = random.choice(sequence_keys)
                    sequence2_id = random.choice(sequence_keys)
                    # Check that the sequences are different
                    while sequence1_id == sequence2_id:
                        sequence2_id = random.choice(sequence_keys)
                    crossOver_sequence = self.generate_crossOver_sequence(
                        self.data.sequences[sequence1_id],
                        self.data.sequences[sequence2_id],
                    )
                    # Add the new sequence to the data object
                    self.data.add_sequence(crossOver_sequence)
                    number_of_sequences = len(self.data.sequences)
                    logging.debug(f"Population size: {number_of_sequences}")

        except Exception as e:
            logging.error(f"Error initializing the population: {e}")

    def generate_mutation_sequence(self, sequence_to_mutate):
        logging.debug("Generating a mutation sequence")
        from Bio.Seq import MutableSeq

        mutable_seq = MutableSeq(sequence_to_mutate)
        for i, aa in enumerate(mutable_seq, start=1):

            if i in self.mutable_positions:
                mutable_seq[i] = random.choice(self.mutable_aa[i])

        mutable_position = random.choice(self.mutable_positions)

    def generate_crossOver_sequence(self):
        logging.debug("Generating a crossOver sequence")

    def get_sequences(self):
        print("Getting the sequences")

    def eval_population(self):
        print("Evaluating the population")

    def crossOver_mutate(self):
        print("Crossing over and mutating the population")
